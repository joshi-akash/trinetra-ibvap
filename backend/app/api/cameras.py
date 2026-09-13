import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from backend.app.config import PROJECT_ROOT
from backend.app.database import get_db
from backend.app.models import CameraRegistry, EntityLog, User
from backend.app.schemas import (
    CameraOut, CameraCreate, CalibrationRequest,
    StreamUpdateRequest, GenericStatusResponse,
    FrameDetectRequest, FrameDetectResponse, DetectedEntityOut
)
from backend.app.auth.jwt_auth import get_current_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])

@router.get("", response_model=List[CameraOut])
def list_cameras(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    cameras = db.query(CameraRegistry).all()
    results = []
    for c in cameras:
        results.append(CameraOut(
            camera_id=c.camera_id,
            location={"lat": c.location_lat, "lon": c.location_lon} if c.location_lat and c.location_lon else None,
            fov_polygon=c.fov_polygon,
            geo_fence_polygon=c.geo_fence_polygon,
            calibration_reference_points=c.calibration_reference_points,
            trust_score=c.trust_score,
            status=c.status,
            last_tamper_check=c.last_tamper_check,
            stream_url=c.stream_url
        ))
    return results

@router.post("", response_model=GenericStatusResponse)
def create_camera(
    req: CameraCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    existing = db.query(CameraRegistry).filter(CameraRegistry.camera_id == req.camera_id).first()
    if existing:
        existing.location_lat = req.location_lat
        existing.location_lon = req.location_lon
        existing.status = req.status
        existing.trust_score = req.trust_score
        if req.geo_fence_polygon:
            existing.geo_fence_polygon = req.geo_fence_polygon
        if req.stream_url is not None:
            existing.stream_url = req.stream_url
        db.commit()
        return GenericStatusResponse(status="success", message=f"Camera {req.camera_id} updated successfully")

    camera = CameraRegistry(
        camera_id=req.camera_id,
        location_lat=req.location_lat,
        location_lon=req.location_lon,
        status=req.status or "online",
        trust_score=req.trust_score or 0.95,
        geo_fence_polygon=req.geo_fence_polygon,
        stream_url=req.stream_url
    )
    db.add(camera)
    db.commit()
    return GenericStatusResponse(status="success", message=f"Camera {req.camera_id} registered successfully")

@router.delete("/{camera_id}", response_model=GenericStatusResponse)
def delete_camera(
    camera_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    try:
        # Nullify foreign key references in EntityLog before deleting camera
        db.query(EntityLog).filter(EntityLog.camera_id == camera_id).update({"camera_id": None})
        db.delete(camera)
        db.commit()
        return GenericStatusResponse(status="success", message=f"Camera {camera_id} removed from registry")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete camera: {str(e)}")


@router.patch("/{camera_id}/status", response_model=GenericStatusResponse)
def toggle_camera_status(
    camera_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    camera.status = "offline" if camera.status == "online" else "online"
    db.commit()
    return GenericStatusResponse(
        status="success",
        message=f"Camera {camera_id} is now {camera.status.upper()}"
    )


def resolve_live_stream_url(url: Optional[str]) -> Optional[str]:
    """
    Resolves a public webcam webpage (e.g. SkylineWebcams) or page link
    into a direct HLS (.m3u8) or MP4 video stream URL.
    """
    if not url:
        return None
    clean_url = url.strip()
    if not clean_url:
        return None

    # 1. Direct video streams require no resolution
    lower_url = clean_url.lower()
    if any(lower_url.endswith(ext) or (ext + "?") in lower_url for ext in [".mp4", ".m3u8", ".webm", ".ogg", ".ts"]):
        return clean_url

    # 2. SkylineWebcams pages (e.g. .../lamai.html)
    if "skylinewebcams.com" in lower_url:
        try:
            import urllib.request
            import re
            req = urllib.request.Request(
                clean_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://www.skylinewebcams.com/"
                }
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            # Extract source:'livee.m3u8?a=...'
            m = re.search(r"source\s*:\s*['\"](livee\.m3u8\?[^'\"]+)['\"]", html)
            if m:
                source = m.group(1).replace("livee.", "live.")
                resolved = f"https://hd-auth.skylinewebcams.com/{source}"
                logger.info(f"Resolved SkylineWebcams page '{clean_url}' -> '{resolved}'")
                return resolved
        except Exception as e:
            logger.warning(f"Failed to resolve SkylineWebcams URL '{clean_url}': {e}")

    return clean_url

@router.put("/{camera_id}/stream", response_model=GenericStatusResponse)
def update_camera_stream(
    camera_id: str,
    req: StreamUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    clean_id = (camera_id or "").strip()
    if not clean_id:
        clean_id = "CAM-01"

    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == clean_id).first()
    raw_url = req.stream_url.strip() if (req.stream_url and req.stream_url.strip()) else None
    new_url = resolve_live_stream_url(raw_url)

    if not camera:
        # Auto-create camera if not already registered
        camera = CameraRegistry(
            camera_id=clean_id,
            location_lat=29.9457,
            location_lon=78.1642,
            status="online",
            trust_score=0.98,
            stream_url=new_url
        )
        db.add(camera)
    else:
        camera.stream_url = new_url

    db.commit()
    msg = f"Stream URL updated for {clean_id}"
    if new_url and new_url != raw_url:
        msg = f"Live HLS stream auto-resolved & updated for {clean_id}"
    return GenericStatusResponse(status="success", message=msg)

@router.post("/{camera_id}/upload-footage", response_model=GenericStatusResponse)
async def upload_camera_footage(
    camera_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    import re
    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == camera_id).first()
    if not camera:
        # Auto-register camera if user uploads to a new camera ID
        camera = CameraRegistry(
            camera_id=camera_id,
            location_lat=29.9457,
            location_lon=78.1642,
            status="online",
            trust_score=0.98,
            stream_url=None
        )
        db.add(camera)

    footage_dir = PROJECT_ROOT / "test_footage"
    footage_dir.mkdir(parents=True, exist_ok=True)
    
    raw_filename = file.filename or f"upload_{camera_id}.mp4"
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', raw_filename)
    dest_file = footage_dir / safe_filename
    with open(dest_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ensure browser-ready H.264 playback via imageio_ffmpeg if available
    final_filename = safe_filename
    try:
        import subprocess, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        base_name = Path(safe_filename).stem
        web_filename = f"{base_name}_web.mp4"
        web_file = footage_dir / web_filename
        cmd = [
            ffmpeg_exe, "-y", "-i", str(dest_file),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            "-movflags", "+faststart",
            str(web_file)
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=30)
        if res.returncode == 0 and web_file.exists() and web_file.stat().st_size > 0:
            final_filename = web_filename
    except Exception:
        final_filename = safe_filename

    camera.stream_url = f"/footage/{final_filename}"
    db.commit()
    return GenericStatusResponse(
        status="success", 
        message=f"Uploaded {file.filename} and bound to {camera_id} (/footage/{final_filename})"
    )

@router.post("/{camera_id}/calibrate", response_model=GenericStatusResponse)
def calibrate_camera(
    camera_id: str,
    req: CalibrationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    points_data = [p.model_dump() for p in req.points]
    camera.calibration_reference_points = {"points": points_data}
    db.commit()

    return GenericStatusResponse(status="success", message=f"Calibration points updated for {camera_id}")

@router.post("/{camera_id}/detect-frame", response_model=FrameDetectResponse)
def detect_camera_frame(
    camera_id: str,
    req: FrameDetectRequest,
    db: Session = Depends(get_db)
):
    """
    Live AI detection endpoint for playing CCTV video footage.
    Runs YOLOv8 detector, ByteTrack tracking, and biometrics on the extracted video frame.
    Returns real bounding boxes and attributes correlated directly with the video objects.
    """
    import base64
    import cv2
    import numpy as np

    raw_b64 = req.image
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    frame_w = req.width or 640
    frame_h = req.height or 360

    try:
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return FrameDetectResponse(camera_id=camera_id, entities=[], frame_width=frame_w, frame_height=frame_h)
    except Exception as e:
        logger.warning(f"Error decoding base64 frame for {camera_id}: {e}")
        return FrameDetectResponse(camera_id=camera_id, entities=[], frame_width=frame_w, frame_height=frame_h)

    h, w = frame.shape[:2]

    try:
        from ai_detection import run_detection_stage
        raw_entities = run_detection_stage(frame, camera_id=camera_id)
    except Exception as e:
        logger.warning(f"Detection stage error: {e}")
        raw_entities = []

    detected_out = []
    for ent in raw_entities:
        bbox = [int(v) for v in ent.get("bbox", [0, 0, 0, 0])]
        attrs = ent.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}

        detected_out.append(DetectedEntityOut(
            entity_type=ent.get("entity_type", "unknown"),
            bbox=bbox,
            confidence=round(float(ent.get("confidence", 0.85)), 2),
            track_id=ent.get("track_id"),
            attributes=attrs
        ))

    return FrameDetectResponse(
        camera_id=camera_id,
        entities=detected_out,
        frame_width=w,
        frame_height=h
    )
