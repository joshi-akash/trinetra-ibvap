"""
Multi-Model Batch Trainer for TRINETRA Surveillance Pipeline.

Trains and tracks progress across all downloaded & curated surveillance models:
1. Overhead & Elevated CCTV Pedestrians (data/cctv_elevated.yaml) -> models/yolov8_custom.pt
2. Perimeter Patrol Vehicles (data/patrol_cam/data.yaml) -> models/yolov8_patrol.pt
3. Tactical Weapons & Melee (data/weapons/data.yaml) -> models/yolov8_weapon.pt
4. LLVIP Night-Vision Humans (data/llvip/data.yaml) -> models/yolov8_night.pt

Provides real-time per-model and overall progress telemetry for the web UI.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("multi_model_trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class ModelProgress:
    id: str
    name: str
    dataset_yaml: str
    target_weights: str
    total_epochs: int
    status: str = "PENDING"  # PENDING, TRAINING, COMPLETED, FAILED
    progress: float = 0.0     # 0.0 to 100.0%
    current_epoch: int = 0
    box_loss: float = 0.0
    cls_loss: float = 0.0
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class MultiModelTrainer:
    """Singleton coordinator for sequential training of all surveillance models."""
    _instance: Optional[MultiModelTrainer] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(MultiModelTrainer, cls).__new__(cls)
                    cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self._thread: Optional[threading.Thread] = None
        self.is_training: bool = False
        self.overall_progress: float = 0.0
        self.active_model_index: int = 0
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.logs: List[str] = []
        self.all_completed: bool = False

        self.models: List[ModelProgress] = [
            ModelProgress(
                id="elevated_cctv",
                name="Elevated CCTV Pedestrians",
                dataset_yaml="data/cctv_elevated.yaml",
                target_weights="models/yolov8_custom.pt",
                total_epochs=10,
            ),
            ModelProgress(
                id="patrol_vehicles",
                name="Perimeter Patrol Vehicles",
                dataset_yaml="data/patrol_cam/data.yaml",
                target_weights="models/yolov8_patrol.pt",
                total_epochs=8,
            ),
            ModelProgress(
                id="tactical_weapons",
                name="Tactical Weapons & Melee",
                dataset_yaml="data/weapons/data.yaml",
                target_weights="models/yolov8_weapon.pt",
                total_epochs=3,
            ),
            ModelProgress(
                id="night_vision",
                name="LLVIP Night-Vision Humans",
                dataset_yaml="data/llvip/data.yaml",
                target_weights="models/yolov8_night.pt",
                total_epochs=3,
            ),
        ]

    def log(self, msg: str):
        logger.info(msg)
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.logs.append(entry)
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            # Recalculate overall progress
            if self.models:
                tot_prog = sum(m.progress for m in self.models)
                self.overall_progress = round(tot_prog / len(self.models), 1)

            active_model = self.models[self.active_model_index] if (0 <= self.active_model_index < len(self.models)) else None

            return {
                "is_training": self.is_training,
                "all_completed": self.all_completed,
                "overall_progress": self.overall_progress,
                "active_model_index": self.active_model_index,
                "total_models": len(self.models),
                "active_model_name": active_model.name if active_model else "",
                "duration_seconds": round(time.time() - self.start_time, 1) if self.is_training else (round(self.end_time - self.start_time, 1) if self.end_time else 0.0),
                "models": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "status": m.status,
                        "progress": round(m.progress, 1),
                        "current_epoch": m.current_epoch,
                        "total_epochs": m.total_epochs,
                        "box_loss": round(m.box_loss, 4),
                        "cls_loss": round(m.cls_loss, 4),
                        "target_weights": m.target_weights,
                        "error": m.error,
                    }
                    for m in self.models
                ],
                "logs": self.logs[-15:],
            }

    def start_batch_training(self) -> bool:
        with self._lock:
            if self.is_training:
                logger.warning("Batch training already running.")
                return False

            self.is_training = True
            self.all_completed = False
            self.overall_progress = 0.0
            self.active_model_index = 0
            self.start_time = time.time()
            self.end_time = 0.0
            self.logs.clear()

            for m in self.models:
                m.status = "PENDING"
                m.progress = 0.0
                m.current_epoch = 0
                m.box_loss = 0.0
                m.cls_loss = 0.0
                m.error = None

            self._thread = threading.Thread(target=self._run_pipeline, daemon=True)
            self._thread.start()
            self.log("Batch training pipeline launched for all 4 surveillance models.")
            return True

    def _run_pipeline(self):
        try:
            from ultralytics import YOLO

            for idx, m in enumerate(self.models):
                with self._lock:
                    self.active_model_index = idx
                    m.status = "TRAINING"
                    m.progress = 5.0
                self.log(f"Starting training on {m.name} ({m.dataset_yaml})...")

                if not os.path.exists(m.dataset_yaml):
                    # Check if LLVIP needs extraction
                    if m.id == "night_vision" and os.path.exists(os.path.expanduser("~/Downloads/LLVIP.zip")):
                        from ai_detection.training.prepare_llvip import prepare_llvip_dataset
                        prepare_llvip_dataset(os.path.expanduser("~/Downloads/LLVIP.zip"))

                if not os.path.exists(m.dataset_yaml):
                    self.log(f"Dataset {m.dataset_yaml} not found. Skipping {m.name}.")
                    with self._lock:
                        m.status = "FAILED"
                        m.error = f"Dataset {m.dataset_yaml} missing"
                        m.progress = 100.0
                    continue

                model_start = time.time()
                try:
                    # Callback to update telemetry on each epoch
                    def on_epoch_end(trainer):
                        ep = trainer.epoch + 1
                        tot = trainer.epochs
                        with self._lock:
                            m.current_epoch = ep
                            m.total_epochs = tot
                            m.progress = min(95.0, round((ep / max(1, tot)) * 100.0, 1))
                            if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                                try:
                                    losses = [float(x) for x in trainer.loss_items]
                                    if len(losses) >= 1:
                                        m.box_loss = losses[0]
                                    if len(losses) >= 2:
                                        m.cls_loss = losses[1]
                                except Exception:
                                    pass
                        self.log(f"[{m.name}] Epoch {ep}/{tot} completed. Loss: {m.box_loss:.4f}")

                    yolo = YOLO("models/yolov8n.pt")
                    yolo.add_callback("on_train_epoch_end", on_epoch_end)

                    imgsz = 320
                    batch = 16 if "weapons" in m.id or "night" in m.id else 8
                    
                    yolo.train(
                        data=m.dataset_yaml,
                        epochs=m.total_epochs,
                        batch=batch,
                        imgsz=imgsz,
                        device="cpu",
                        project="runs/train",
                        name=f"batch_{m.id}",
                        exist_ok=True,
                        pretrained=True,
                        optimizer="AdamW",
                        lr0=0.001,
                        lrf=0.01,
                        verbose=False,
                    )

                    # Locate best.pt
                    best_pt = os.path.join("runs", "train", f"batch_{m.id}", "weights", "best.pt")
                    if not os.path.exists(best_pt):
                        user_runs = os.path.expanduser(os.path.join("~", "runs", "detect", "runs", "train", f"batch_{m.id}", "weights", "best.pt"))
                        if os.path.exists(user_runs):
                            best_pt = user_runs

                    if os.path.exists(best_pt):
                        os.makedirs(os.path.dirname(m.target_weights), exist_ok=True)
                        shutil.copy(best_pt, m.target_weights)
                        self.log(f"Saved {m.name} to {m.target_weights}")

                    with self._lock:
                        m.status = "COMPLETED"
                        m.progress = 100.0
                        m.elapsed_seconds = round(time.time() - model_start, 1)

                    self.log(f"Successfully trained {m.name} in {m.elapsed_seconds}s!")

                except Exception as e:
                    self.log(f"Training error on {m.name}: {e}")
                    with self._lock:
                        m.status = "FAILED"
                        m.error = str(e)
                        m.progress = 100.0

            # Hot reload the custom model
            try:
                from ai_detection.training import get_auto_trainer
                trainer = get_auto_trainer()
                trainer.apply_weights("models/yolov8_custom.pt")
                self.log("Active TRINETRA fleet successfully hot-reloaded with refined custom weights!")
            except Exception as e:
                self.log(f"Hot reload notice: {e}")

            with self._lock:
                self.is_training = False
                self.all_completed = True
                self.end_time = time.time()
                self.overall_progress = 100.0

            self.log(f"BATCH RETRAINING COMPLETE: All surveillance models trained and deployed!")

        except Exception as ex:
            logger.error(f"Batch training failed: {ex}")
            with self._lock:
                self.is_training = False
                self.all_completed = False
                self.end_time = time.time()
            self.log(f"Fatal error in batch training: {ex}")


_batch_trainer_instance: Optional[MultiModelTrainer] = None


def get_multi_trainer() -> MultiModelTrainer:
    global _batch_trainer_instance
    if _batch_trainer_instance is None:
        _batch_trainer_instance = MultiModelTrainer()
    return _batch_trainer_instance
