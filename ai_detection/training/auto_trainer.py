"""
Continuous Model Auto-Trainer for TRINETRA CCTV Platform.

Enables operators to upload CCTV video files (MP4, MKV, AVI) or provide live video stream
links (YouTube Live, HLS m3u8, RTSP, SkylineWebcams) to:
1. Ingest and resolve video stream / local file.
2. Sample keyframes with optional low-light (Zero-DCE / CLAHE) enhancement.
3. Automatically pseudo-label frames using the existing high-confidence YOLOv8 detector.
4. Structure a train/val dataset under data/custom_dataset/.
5. Fine-tune YOLOv8 using Ultralytics with real-time epoch telemetry and loss tracking.
6. Export custom trained weights to models/yolov8_custom.pt.
7. Hot-reload the active detection engine with the refined weights.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("auto_trainer")

# Standard 80 COCO classes for seamless YOLOv8 head compatibility
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


def resolve_stream_source(url_or_path: str) -> str:
    """
    Resolve a video stream URL or file path.
    Supports YouTube URLs (using yt-dlp), SkylineWebcams, direct HLS/RTSP/MP4, and local files.
    """
    if not url_or_path:
        return ""
    clean = url_or_path.strip()
    if os.path.isfile(clean):
        return os.path.abspath(clean)

    lower = clean.lower()

    # 1. YouTube link handling via yt-dlp
    if "youtube.com" in lower or "youtu.be" in lower:
        try:
            import yt_dlp
            ydl_opts = {
                "format": "best[ext=mp4]/best",
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean, download=False)
                if "url" in info:
                    logger.info(f"Resolved YouTube link '{clean}' via yt-dlp to stream URL")
                    return info["url"]
                elif "formats" in info and len(info["formats"]) > 0:
                    return info["formats"][-1]["url"]
        except Exception as e:
            logger.warning(f"yt-dlp stream resolution failed for '{clean}': {e}")
            return clean

    # 2. SkylineWebcams link handling
    if "skylinewebcams.com" in lower:
        try:
            import urllib.request
            req = urllib.request.Request(
                clean,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Referer": "https://www.skylinewebcams.com/",
                }
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            m = re.search(r"source\s*:\s*['\"](livee\.m3u8\?[^'\"]+)['\"]", html)
            if m:
                source = m.group(1).replace("livee.", "live.")
                resolved = f"https://hd-auth.skylinewebcams.com/{source}"
                logger.info(f"Resolved SkylineWebcams page '{clean}' -> '{resolved}'")
                return resolved
        except Exception as e:
            logger.warning(f"SkylineWebcams resolution error for '{clean}': {e}")

    return clean


@dataclass
class TrainingState:
    status: str = "IDLE"  # IDLE, INGESTING, EXTRACTING_FRAMES, AUTO_ANNOTATING, TRAINING_EPOCHS, COMPLETED, FAILED
    progress: float = 0.0  # 0.0 to 100.0%
    current_epoch: int = 0
    total_epochs: int = 0
    total_frames: int = 0
    annotated_boxes: int = 0
    box_loss: float = 0.0
    cls_loss: float = 0.0
    weights_path: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


class AutoTrainer:
    """Singleton trainer managing background continual model fine-tuning."""
    _instance: Optional[AutoTrainer] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(AutoTrainer, cls).__new__(cls)
                    cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self.state = TrainingState()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False

    def log(self, message: str):
        logger.info(message)
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.state.logs.append(entry)
        if len(self.state.logs) > 150:
            self.state.logs = self.state.logs[-150:]

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.state.status,
                "progress": round(self.state.progress, 1),
                "current_epoch": self.state.current_epoch,
                "total_epochs": self.state.total_epochs,
                "total_frames": self.state.total_frames,
                "annotated_boxes": self.state.annotated_boxes,
                "box_loss": round(self.state.box_loss, 4),
                "cls_loss": round(self.state.cls_loss, 4),
                "weights_path": self.state.weights_path,
                "error": self.state.error,
                "logs": self.state.logs,
                "duration_seconds": round(time.time() - self.state.start_time, 1) if self.state.start_time and self.state.status != "COMPLETED" else round(self.state.end_time - self.state.start_time, 1) if self.state.end_time else 0.0,
            }

    def start_training(
        self,
        source_type: str,  # 'file' or 'stream'
        source_path_or_url: str,
        epochs: int = 3,
        max_frames: int = 40,
        base_weights: str = "models/yolov8n.pt",
        enhance_low_light: bool = True,
        dataset_dir: str = "data/custom_dataset",
        output_weights: str = "models/yolov8_custom.pt",
    ) -> bool:
        """Start background training thread if not already running."""
        with self._lock:
            if self.state.status in ["INGESTING", "EXTRACTING_FRAMES", "AUTO_ANNOTATING", "TRAINING_EPOCHS"]:
                return False

            self.state = TrainingState(
                status="INGESTING",
                progress=2.0,
                total_epochs=max(1, epochs),
                start_time=time.time(),
            )
            self._stop_requested = False

        self._thread = threading.Thread(
            target=self._training_worker,
            args=(source_type, source_path_or_url, epochs, max_frames, base_weights, enhance_low_light, dataset_dir, output_weights),
            daemon=True,
        )
        self._thread.start()
        return True

    def _training_worker(
        self,
        source_type: str,
        source_path_or_url: str,
        epochs: int,
        max_frames: int,
        base_weights: str,
        enhance_low_light: bool,
        dataset_dir: str,
        output_weights: str,
    ):
        try:
            self.log(f"Starting TRINETRA Continual Training Pipeline [Source: {source_type}]")
            self.log(f"Config: {epochs} epochs | max {max_frames} frames | base weights: {base_weights}")

            # 1. Resolve Source
            self.state.status = "INGESTING"
            self.state.progress = 5.0
            resolved_source = resolve_stream_source(source_path_or_url)
            self.log(f"Ingested source: {resolved_source[:80]}...")

            # 2. Extract Keyframes
            self.state.status = "EXTRACTING_FRAMES"
            self.state.progress = 10.0

            dataset_path = Path(dataset_dir).resolve()
            train_img_dir = dataset_path / "images" / "train"
            val_img_dir = dataset_path / "images" / "val"
            train_lbl_dir = dataset_path / "labels" / "train"
            val_lbl_dir = dataset_path / "labels" / "val"

            # Clean/create directories
            for p in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
                p.mkdir(parents=True, exist_ok=True)
                for f in p.glob("*.*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass

            cap = cv2.VideoCapture(resolved_source)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {resolved_source}")

            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.log(f"Video stream opened. Total frames: {total_video_frames}, FPS: {fps:.1f}")

            extracted_frames = []
            if total_video_frames > 0:
                # Video file: sample evenly
                step = max(1, total_video_frames // max_frames)
                frame_idx = 0
                while cap.isOpened() and len(extracted_frames) < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % step == 0:
                        extracted_frames.append(frame)
                    frame_idx += 1
            else:
                # Live stream: sample with short sleep interval
                read_count = 0
                max_attempts = max_frames * 10
                while cap.isOpened() and len(extracted_frames) < max_frames and read_count < max_attempts:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        read_count += 1
                        continue
                    if read_count % 8 == 0:
                        extracted_frames.append(frame)
                    read_count += 1
                    time.sleep(0.02)

            cap.release()

            if not extracted_frames:
                raise RuntimeError("Zero valid frames extracted from video source.")

            self.state.total_frames = len(extracted_frames)
            self.log(f"Extracted {len(extracted_frames)} keyframes for training dataset.")
            self.state.progress = 25.0

            # 3. Optional Low-Light Enhancement & Save
            saved_paths = []
            for i, frame in enumerate(extracted_frames):
                # Enhance low light if enabled
                if enhance_low_light:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_lum = float(np.mean(gray))
                    if mean_lum < 55.0:
                        # CLAHE enhancement
                        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        l_enh = clahe.apply(l)
                        enhanced_lab = cv2.merge((l_enh, a, b))
                        frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

                is_val = (i % 5 == 0)  # 20% validation split
                target_dir = val_img_dir if is_val else train_img_dir
                img_name = f"cctv_frame_{i:04d}.jpg"
                img_path = target_dir / img_name
                cv2.imwrite(str(img_path), frame)
                saved_paths.append((img_path, is_val, frame.shape))

            self.state.progress = 35.0

            # 4. Auto-Annotation (Pseudo-Labeling)
            self.state.status = "AUTO_ANNOTATING"
            self.log("Running YOLO detector for automated pseudo-annotation...")

            from ultralytics import YOLO

            # Load base weights for annotation
            annotation_weights = base_weights if os.path.exists(base_weights) else "models/yolov8s.pt"
            if not os.path.exists(annotation_weights):
                annotation_weights = "yolov8s.pt"

            annotator_model = YOLO(annotation_weights)
            total_boxes = 0

            for i, (img_path, is_val, (h, w, _)) in enumerate(saved_paths):
                lbl_dir = val_lbl_dir if is_val else train_lbl_dir
                lbl_path = lbl_dir / f"{img_path.stem}.txt"

                results = annotator_model.predict(str(img_path), conf=0.35, verbose=False)
                lines = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        xywhn = box.xywhn[0].tolist()  # [x_center, y_center, width, height] normalized
                        lines.append(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                        total_boxes += 1

                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

                self.state.progress = 35.0 + (15.0 * (i + 1) / len(saved_paths))

            self.state.annotated_boxes = total_boxes
            self.log(f"Auto-annotation complete: generated {total_boxes} bounding boxes across {len(saved_paths)} frames.")
            self.state.progress = 50.0

            # 5. Generate dataset.yaml
            yaml_path = dataset_path / "dataset.yaml"
            names_dict = {i: name for i, name in enumerate(COCO_CLASSES)}
            names_yaml_lines = [f"  {i}: {name}" for i, name in names_dict.items()]

            yaml_content = f"""path: '{dataset_path.as_posix()}'
train: 'images/train'
val: 'images/val'
names:
{chr(10).join(names_yaml_lines)}
"""
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            self.log(f"Generated dataset specification: {yaml_path}")
            self.state.progress = 55.0

            # 6. Fine-Tuning Execution
            self.state.status = "TRAINING_EPOCHS"
            self.log(f"Launching Ultralytics fine-tuning for {epochs} epochs...")

            # Select device
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
            self.log(f"Using computing hardware: {device.upper()} (PyTorch {torch.__version__})")

            model = YOLO(annotation_weights)

            # Define telemetry callback
            def on_train_epoch_end(trainer):
                try:
                    ep = trainer.epoch + 1
                    self.state.current_epoch = ep
                    # Safely extract losses from dictionary or tensor array
                    losses = getattr(trainer, "loss_items", None)
                    if losses is not None:
                        if isinstance(losses, dict):
                            self.state.box_loss = float(losses.get("box_loss", losses.get("box", 0.0)))
                            self.state.cls_loss = float(losses.get("cls_loss", losses.get("cls", 0.0)))
                        elif hasattr(losses, "__iter__"):
                            l_items = list(losses)
                            if len(l_items) > 0:
                                self.state.box_loss = float(l_items[0])
                            if len(l_items) > 1:
                                self.state.cls_loss = float(l_items[1])
                except Exception as e:
                    logger.debug("Loss callback parsing error: %s", e)

                # Calculate progress from 55% to 95%
                epoch_prog = 55.0 + (40.0 * ep / max(1, epochs))
                self.state.progress = min(95.0, epoch_prog)
                self.log(f"Epoch {ep}/{epochs} finished - Box Loss: {self.state.box_loss:.4f} | Cls Loss: {self.state.cls_loss:.4f}")

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            runs_dir = Path("runs/train").resolve()
            runs_dir.mkdir(parents=True, exist_ok=True)

            model.train(
                data=str(yaml_path),
                epochs=epochs,
                batch=4,
                imgsz=640,
                device=device,
                project=str(runs_dir),
                name="cctv_custom_run",
                exist_ok=True,
                pretrained=True,
                optimizer="AdamW",
                lr0=0.001,
                verbose=False,
            )

            # 7. Locate & Save Best Trained Weights
            best_weights = runs_dir / "cctv_custom_run" / "weights" / "best.pt"
            last_weights = runs_dir / "cctv_custom_run" / "weights" / "last.pt"
            src_weights = best_weights if best_weights.exists() else last_weights

            out_path = Path(output_weights).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if src_weights.exists():
                shutil.copy2(src_weights, out_path)
                self.log(f"Successfully saved refined weights to: {out_path}")
            else:
                # Save the trained model directly
                model.save(str(out_path))
                self.log(f"Model saved to: {out_path}")

            self.state.weights_path = str(out_path)
            self.state.status = "COMPLETED"
            self.state.progress = 100.0
            self.state.end_time = time.time()
            self.log(f"🎉 Pipeline completed successfully in {self.state.end_time - self.state.start_time:.1f}s. Model ready for deployment.")

        except Exception as e:
            logger.exception("Training pipeline failed: %s", e)
            self.state.status = "FAILED"
            self.state.error = str(e)
            self.state.end_time = time.time()
            self.log(f"❌ Training error: {e}")

    def apply_weights(self, weights_path: Optional[str] = None) -> bool:
        """Hot-reload the active AI detector with the trained weights."""
        target_path = weights_path or self.state.weights_path or "models/yolov8_custom.pt"
        if not os.path.exists(target_path):
            self.log(f"Weights file not found: {target_path}")
            return False

        try:
            from ai_detection import reload_detector
            reload_detector(model_path=target_path)
            self.log(f"Active AI detector successfully hot-reloaded with: {target_path}")
            return True
        except Exception as e:
            self.log(f"Failed to reload detector: {e}")
            return False


# Global singleton
_auto_trainer_instance: Optional[AutoTrainer] = None


def get_auto_trainer() -> AutoTrainer:
    """Retrieve global AutoTrainer instance."""
    global _auto_trainer_instance
    if _auto_trainer_instance is None:
        _auto_trainer_instance = AutoTrainer()
    return _auto_trainer_instance
