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

    def _get_camera_video_source(self, camera_id: str) -> Optional[Path]:
        """Resolves local CCTV footage file bound to a camera registry entry."""
        try:
            from backend.app.database import SessionLocal
            from backend.app.models import CameraRegistry
            from backend.app.config import PROJECT_ROOT
            with SessionLocal() as db:
                cam = db.query(CameraRegistry).filter(CameraRegistry.camera_id == camera_id).first()
                if cam and cam.stream_url:
                    su = cam.stream_url.replace("\\", "/")
                    if su.startswith("/footage/"):
                        p = PROJECT_ROOT / "test_footage" / su[len("/footage/"):]
                        if p.exists() and p.stat().st_size > 0:
                            return p
                    direct_p = Path(cam.stream_url)
                    if direct_p.is_absolute() and direct_p.exists() and direct_p.stat().st_size > 0:
                        return direct_p
                    rel_p = PROJECT_ROOT / cam.stream_url.lstrip("/")
                    if rel_p.exists() and rel_p.stat().st_size > 0:
                        return rel_p
        except Exception:
            pass
        return None

    def extract_30s_clip(self, camera_id: str, trigger_time: datetime, alert_id: str) -> str:
        """
        Extracts evidence clip from camera source video or ring buffer:
        - In cases where total clip is <= 30 seconds: simply gives the whole clip as output!
        - In cases where total clip is > 30 seconds: extracts an exact 30-second evidence window.
        - Ring buffer: outputs all available frames without dummy placeholder padding.
        """
        clip_filename = f"{alert_id}.mp4"
        clip_path = settings.CLIPS_DIR / clip_filename

        # A. Check if camera has a real video file bound
        src_video = self._get_camera_video_source(camera_id)
        if src_video and src_video.exists() and src_video.stat().st_size > 0:
            try:
                import cv2
                import shutil
                cap = cv2.VideoCapture(str(src_video))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or not (1 <= fps <= 120):
                    fps = 25.0
                duration_sec = total_frames / fps if total_frames > 0 else 0.0
                cap.release()

                # User requirement: when total clip is <= 30 seconds, give the whole clip as output
                if duration_sec <= 30.0 or duration_sec == 0.0:
                    shutil.copyfile(str(src_video), str(clip_path))
                    return str(clip_path)

                # Total clip is > 30 seconds: Extract a clean 30-second window via FFmpeg
                try:
                    import subprocess, imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    start_sec = 0.0
                    cmd = [
                        ffmpeg_exe, "-y",
                        "-ss", str(start_sec),
                        "-i", str(src_video),
                        "-t", "30.0",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "fast",
                        "-movflags", "+faststart",
                        str(clip_path)
                    ]
                    res = subprocess.run(cmd, capture_output=True, timeout=30)
                    if res.returncode == 0 and clip_path.exists() and clip_path.stat().st_size > 0:
                        return str(clip_path)
                except Exception:
                    pass

                # Fallback if FFmpeg slicing fails: give the whole clip as output
                shutil.copyfile(str(src_video), str(clip_path))
                return str(clip_path)
            except Exception:
                pass

        # B. Ingest from Ring Buffer
        buf = self.get_buffer(camera_id)
        start_time = trigger_time - timedelta(seconds=settings.PRE_TRIGGER_DURATION_SEC)
        end_time = trigger_time + timedelta(seconds=settings.POST_TRIGGER_DURATION_SEC)
        frames = buf.get_frames_in_window(start_time, end_time)

        try:
            import cv2
            import numpy as np

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(clip_path), fourcc, 10.0, (640, 480))
            if frames:
                # Output all available real frames in window (even if buffer duration is < 30s)
                for f in frames:
                    if f.image_bytes:
                        nparr = np.frombuffer(f.image_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            if (img.shape[1], img.shape[0]) != (640, 480):
                                img = cv2.resize(img, (640, 480))
                            out.write(img)
            else:
                for i in range(20):
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(dummy, f"TRINETRA REPLAY CAM {camera_id} T+{i*0.1:.1f}s", (30, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    out.write(dummy)
            out.release()
        except Exception:
            with open(clip_path, "wb") as f:
                header = f"TRINETRA_EVIDENCE_CLIP|CAM={camera_id}|TRIGGER={trigger_time.isoformat()}|FRAMES={len(frames)}\n"
                f.write(header.encode('utf-8'))
                for frame in frames:
                    if frame.image_bytes:
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
        Uses ring buffer frame or camera video footage frame.
        Automatically applies night enhancement (contrast/brightness boost) in low-light conditions.
        """
        buf = self.get_buffer(camera_id)
        with buf.lock:
            latest_frame = buf.buffer[-1] if buf.buffer else None

        thumb_filename = f"{alert_id}.webp"
        thumb_path = settings.THUMBNAILS_DIR / thumb_filename

        img = None
        if latest_frame and latest_frame.image_bytes:
            try:
                img = Image.open(io.BytesIO(latest_frame.image_bytes))
            except Exception:
                img = None

        # If ring buffer didn't have a frame, try to extract a real frame from the camera's source video
        if img is None:
            src_video = self._get_camera_video_source(camera_id)
            if src_video and src_video.exists():
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(src_video))
                    ret, frame_bgr = cap.read()
                    cap.release()
                    if ret and frame_bgr is not None:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                except Exception:
                    img = None

        if img is None:
            img = Image.new("RGB", (640, 480), color=(30, 35, 45))
            draw = ImageDraw.Draw(img)
            draw.text((50, 220), f"ALERT {alert_id} - CAM {camera_id}", fill=(255, 50, 50))

        if is_low_light:
            try:
                from PIL import ImageEnhance
                img = ImageEnhance.Brightness(img).enhance(1.6)
                img = ImageEnhance.Contrast(img).enhance(1.4)
            except Exception:
                pass

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
