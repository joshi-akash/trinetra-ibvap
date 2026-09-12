import time
import io
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from backend.app.config import settings

class FrameRecord:
    def __init__(self, timestamp: datetime, frame_id: str, image_bytes: Optional[bytes] = None):
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.image_bytes = image_bytes or self._generate_placeholder_frame(frame_id, timestamp)

    def _generate_placeholder_frame(self, frame_id: str, timestamp: datetime) -> bytes:
        img = Image.new("RGB", (640, 480), color=(30, 35, 45))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"TRINETRA CCTV - {timestamp.isoformat()}", fill=(200, 220, 255))
        draw.text((20, 50), f"FRAME: {frame_id}", fill=(180, 180, 180))
        draw.rectangle([200, 150, 440, 350], outline=(0, 255, 128), width=2)
        draw.text((205, 155), "TARGET REGION", fill=(0, 255, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()


class CameraRingBuffer:
    """Rolling ~60-second circular buffer for a single camera."""
    def __init__(self, camera_id: str, capacity_seconds: int = 60, fps: int = 10):
        self.camera_id = camera_id
        self.capacity_seconds = capacity_seconds
        self.fps = fps
        self.max_frames = capacity_seconds * fps
        self.buffer: deque[FrameRecord] = deque(maxlen=self.max_frames)
        self.lock = threading.Lock()

    def add_frame(self, frame_id: str, timestamp: Optional[datetime] = None, image_bytes: Optional[bytes] = None) -> FrameRecord:
        ts = timestamp or datetime.now(timezone.utc)
        record = FrameRecord(timestamp=ts, frame_id=frame_id, image_bytes=image_bytes)
        with self.lock:
            self.buffer.append(record)
        return record

    def get_frames_in_window(self, start_time: datetime, end_time: datetime) -> List[FrameRecord]:
        with self.lock:
            return [f for f in self.buffer if start_time <= f.timestamp <= end_time]

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)


class RTSPIngestionService:
    """Manages multi-camera RTSP ingestion and evidence extraction."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RTSPIngestionService, cls).__new__(cls)
                cls._instance._buffers: Dict[str, CameraRingBuffer] = {}
        return cls._instance

    def get_buffer(self, camera_id: str) -> CameraRingBuffer:
        if camera_id not in self._buffers:
            self._buffers[camera_id] = CameraRingBuffer(
                camera_id=camera_id,
                capacity_seconds=settings.RING_BUFFER_DURATION_SEC
            )
        return self._buffers[camera_id]

    def push_frame(self, camera_id: str, frame_id: str, timestamp: Optional[datetime] = None, image_bytes: Optional[bytes] = None) -> FrameRecord:
        buf = self.get_buffer(camera_id)
        return buf.add_frame(frame_id=frame_id, timestamp=timestamp, image_bytes=image_bytes)

    def extract_30s_clip(self, camera_id: str, trigger_time: datetime, alert_id: str) -> str:
        """
        Extracts 30-second evidence clip (15s pre-trigger + 15s post-trigger) from ring buffer.
        Saves clip to local storage and returns the relative/absolute file path.
        """
        buf = self.get_buffer(camera_id)
        start_time = trigger_time - timedelta(seconds=settings.PRE_TRIGGER_DURATION_SEC)
        end_time = trigger_time + timedelta(seconds=settings.POST_TRIGGER_DURATION_SEC)
        frames = buf.get_frames_in_window(start_time, end_time)

        clip_filename = f"{alert_id}.mp4"
        clip_path = settings.CLIPS_DIR / clip_filename

        # Write clip metadata and raw keyframe sequence bundle or MP4
        # If openCV is available, we write a valid MP4; otherwise save evidence package
        try:
            import cv2
            import numpy as np

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(clip_path), fourcc, 10.0, (640, 480))
            if frames:
                for f in frames:
                    nparr = np.frombuffer(f.image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        if (img.shape[1], img.shape[0]) != (640, 480):
                            img = cv2.resize(img, (640, 480))
                        out.write(img)
            else:
                # Create fallback frames if buffer had no prior frames
                for i in range(20):
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(dummy, f"TRINETRA REPLAY CAM {camera_id} T+{i*0.1:.1f}s", (30, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    out.write(dummy)
            out.release()
        except Exception:
            # Fallback for environments without OpenCV video codec support: save binary stream file
            with open(clip_path, "wb") as f:
                header = f"TRINETRA_EVIDENCE_CLIP|CAM={camera_id}|TRIGGER={trigger_time.isoformat()}|FRAMES={len(frames)}\n"
                f.write(header.encode('utf-8'))
                for frame in frames:
                    f.write(frame.image_bytes)

        return str(clip_path)

    def extract_thumbnail(
        self,
        camera_id: str,
        trigger_time: datetime,
        alert_id: str,
        bbox: Optional[List[float]] = None,
        is_low_light: bool = False
    ) -> str:
        """
        Extracts and saves a WebP thumbnail around the trigger event.
        Automatically applies night enhancement (contrast/brightness boost) in low-light conditions.
        """
        buf = self.get_buffer(camera_id)
        with buf.lock:
            latest_frame = buf.buffer[-1] if buf.buffer else None

        thumb_filename = f"{alert_id}.webp"
        thumb_path = settings.THUMBNAILS_DIR / thumb_filename

        if latest_frame and latest_frame.image_bytes:
            img = Image.open(io.BytesIO(latest_frame.image_bytes))
            if is_low_light:
                try:
                    from PIL import ImageEnhance
                    img = ImageEnhance.Brightness(img).enhance(1.6)
                    img = ImageEnhance.Contrast(img).enhance(1.4)
                except Exception:
                    pass
        else:
            img = Image.new("RGB", (640, 480), color=(40, 20, 20))
            draw = ImageDraw.Draw(img)
            draw.text((50, 220), f"ALERT {alert_id} - CAM {camera_id}", fill=(255, 50, 50))

        # Annotate bbox and threat target
        draw = ImageDraw.Draw(img)
        if bbox and len(bbox) == 4:
            draw.rectangle(bbox, outline=(255, 34, 68), width=3)
            draw.text((bbox[0], max(0, bbox[1] - 15)), "THREAT TARGET", fill=(255, 34, 68))

        if is_low_light:
            draw.text((20, 20), "🌙 NIGHT SECTOR / LOW LIGHT", fill=(0, 229, 255))

        # Save as compressed WebP
        img.save(thumb_path, format="WEBP", quality=85)
        return str(thumb_path)


rtsp_service = RTSPIngestionService()
