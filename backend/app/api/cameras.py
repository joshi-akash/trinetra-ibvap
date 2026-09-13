import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

import time
import uuid
from datetime import datetime, timezone
from backend.app.config import PROJECT_ROOT, settings
from backend.app.database import get_db
from backend.app.models import CameraRegistry, EntityLog, User
from backend.app.rule_engine.geofence_check import geofence_evaluator
from backend.app.schemas import (
    CameraOut, CameraCreate, CalibrationRequest,
    StreamUpdateRequest, GenericStatusResponse,
    FrameDetectRequest, FrameDetectResponse, DetectedEntityOut
)
from backend.app.auth.jwt_auth import get_current_user
import logging

logger = logging.getLogger(__name__)

# Global rate-limiting cache to throttle database writes per (camera_id, track_id, is_alert)
_ENTITY_LOG_THROTTLE: dict = {}

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

    # 2. YouTube streams (youtube.com or youtu.be)
    if "youtube.com" in lower_url or "youtu.be" in lower_url:
        try:
            import yt_dlp
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=False)
                if info:
                    if info.get("hls_url"):
                        logger.info(f"Resolved YouTube hls_url '{clean_url}'")
                        return info["hls_url"]
                    if info.get("manifest_url"):
                        logger.info(f"Resolved YouTube manifest_url '{clean_url}'")
                        return info["manifest_url"]
                    fmts = info.get("formats", [])
                    m3u8_fmts = [f for f in fmts if "m3u8" in f.get("protocol", "") and f.get("vcodec") != "none"]
                    if m3u8_fmts:
                        logger.info(f"Resolved YouTube m3u8 format '{clean_url}'")
                        return m3u8_fmts[-1].get("url")
                    mp4_fmts = [f for f in fmts if f.get("vcodec") != "none" and f.get("url") and ("http" in f.get("protocol", "") or f.get("ext") == "mp4")]
                    if mp4_fmts:
                        logger.info(f"Resolved YouTube mp4 format '{clean_url}'")
                        return mp4_fmts[-1].get("url")
                    if info.get("url"):
                        return info["url"]
        except Exception as e:
            logger.warning(f"Failed to resolve YouTube URL '{clean_url}': {e}")

    # 3. SkylineWebcams pages (e.g. .../lamai.html)
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

    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == camera_id).first()
    geofence_coords = camera.geo_fence_polygon if camera else None
    cam_lat = camera.location_lat if (camera and camera.location_lat) else 29.9457
    cam_lon = camera.location_lon if (camera and camera.location_lon) else 78.1642

    detected_out = []
    now_ts = time.time()
    thumb_dir = Path(settings.STORAGE_DIR) / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    for ent in raw_entities:
        bbox = [int(v) for v in ent.get("bbox", [0, 0, 0, 0])]
        attrs = ent.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}

        raw_tid = ent.get("track_id")
        tid_val = None
        if raw_tid is not None:
            try:
                tid_val = int(raw_tid)
            except (ValueError, TypeError):
                tid_val = str(raw_tid)

        ent_type = ent.get("entity_type", "unknown")
        conf_val = round(float(ent.get("confidence", 0.85)), 2)

        # 1. Coordinate estimation & Footpoint
        x_foot = (bbox[0] + bbox[2]) / 2.0
        y_foot = float(bbox[3])
        ent_lat = cam_lat + ((360 - y_foot) / 360.0) * 0.0008
        ent_lon = cam_lon + ((x_foot - 320) / 640.0) * 0.0008

        # 2. Geofence evaluation (pixel or geographic coordinates)
        is_breach = False
        if geofence_coords and len(geofence_coords) >= 3:
            first_pt = geofence_coords[0]
            if first_pt[0] > 180 or first_pt[0] < -180:
                is_breach = geofence_evaluator.bbox_intersects_geofence(bbox, None, geofence_coords)
            else:
                is_breach = geofence_evaluator.is_inside_geofence(ent_lat, ent_lon, geofence_coords)

        # 3. Behavioral and Prop threat checks
        props = attrs.get("props", [])
        if not isinstance(props, list):
            props = []

        posture = (attrs.get("posture") or "standing").lower().strip()

        # Weapon possession check (illegal weapon)
        has_weapon = any(w in props for w in ["weapon", "knife", "gun", "rifle", "firearm"])

        # Big bag carrying check (unattended or suspicious large backpack / suitcase)
        has_big_bag = any(b in props for b in ["large_backpack", "backpack", "suitcase", "handbag", "big_bag"])

        # Suspicious body posture check (crouching, prone/crawling, climbing)
        has_sus_posture = posture in ("crouching", "prone", "crawling", "climbing", "sprinting")

        # Movement anomaly check (sprinting, loitering, sudden direction reversal)
        movement_flags = attrs.get("movement_flags", [])
        if not isinstance(movement_flags, list):
            movement_flags = []
        has_movement_anomaly = bool(movement_flags)

        # Suspect face match check
        face_name = attrs.get("face_name")
        is_suspect = bool(face_name and face_name != "Unidentified")

        # 4. Decision Bifurcation Logic
        is_alert = False
        alert_type = None
        rule_fired = None
        threat_score = 0.10

        if has_weapon and (is_breach or has_sus_posture):
            is_alert = True
            alert_type = "correlated"
            rule_fired = f"multi_modal_corroboration: illegal_weapon & {'perimeter_breach' if is_breach else 'suspicious_posture'}"
            threat_score = 0.98
        elif has_weapon:
            is_alert = True
            alert_type = "behavior"
            rule_fired = "threat_prop: weapon (illegal weapon possession)"
            threat_score = 0.95
        elif has_big_bag and is_breach:
            is_alert = True
            alert_type = "correlated"
            rule_fired = "multi_modal_corroboration: suspicious_bag & perimeter_breach"
            threat_score = 0.90
        elif has_big_bag:
            is_alert = True
            alert_type = "behavior"
            rule_fired = "threat_prop: large_backpack (suspicious big bag / baggage)"
            threat_score = 0.85
        elif has_movement_anomaly and is_breach:
            is_alert = True
            alert_type = "correlated"
            rule_fired = f"multi_modal_corroboration: breach & {movement_flags[0]}"
            threat_score = 0.93
        elif has_movement_anomaly:
            is_alert = True
            alert_type = "behavior"
            rule_fired = f"suspicious_movement: {', '.join(movement_flags)}"
            threat_score = 0.86
        elif has_sus_posture and is_breach:
            is_alert = True
            alert_type = "correlated"
            rule_fired = f"multi_modal_corroboration: breach & {posture}_posture"
            threat_score = 0.92
        elif has_sus_posture:
            is_alert = True
            alert_type = "behavior"
            rule_fired = f"suspicious_posture: {posture}"
            threat_score = 0.82 if posture == "crouching" else 0.88
        elif is_breach:
            is_alert = True
            alert_type = "geo_fence"
            rule_fired = "geofence_polygon_breach"
            threat_score = 0.88
        elif is_suspect:
            is_alert = True
            alert_type = "behavior"
            rule_fired = f"suspect_face_match: {face_name}"
            threat_score = 0.92

        # 5. Event-Driven Intelligent Database Persistence & Throttling
        # Rules:
        # a) First sighting of a tracked entity -> Log once immediately.
        # b) New / Escalated Alert -> Log immediately with thumbnail.
        # c) Repeated Alert (same track + same rule) -> Cooldown for at least 45 seconds.
        # d) Forensic Milestone -> Log if license plate OCR or suspect face newly identified.
        # e) Passive heartbeat -> At most once every 60 seconds.
        # f) Untracked entity -> Cooldown 30s (passive) or 15s (alert).
        track_state_key = f"{camera_id}_{tid_val}" if tid_val is not None else f"{camera_id}_untracked_{is_alert}"
        track_state = _ENTITY_LOG_THROTTLE.get(track_state_key)

        should_log = False
        if track_state is None:
            should_log = True
            _ENTITY_LOG_THROTTLE[track_state_key] = {
                "first_seen": now_ts,
                "last_logged": now_ts,
                "last_rule": rule_fired,
                "has_alerted": is_alert,
                "plate": attrs.get("plate_text"),
                "face": attrs.get("face_name")
            }
        else:
            time_since_log = now_ts - track_state["last_logged"]
            if is_alert:
                # Log immediately if rule escalated/changed (e.g. loitering -> weapon or breach)
                if rule_fired != track_state.get("last_rule"):
                    should_log = True
                    track_state["last_rule"] = rule_fired
                    track_state["last_logged"] = now_ts
                    track_state["has_alerted"] = True
                elif time_since_log >= 45.0:
                    should_log = True
                    track_state["last_logged"] = now_ts
            else:
                # Passive entity: check for newly identified license plate or suspect face
                new_plate = bool(attrs.get("plate_text") and attrs.get("plate_text") != track_state.get("plate"))
                new_face = bool(attrs.get("face_name") and attrs.get("face_name") != "Unidentified" and attrs.get("face_name") != track_state.get("face"))
                if new_plate or new_face:
                    should_log = True
                    track_state["plate"] = attrs.get("plate_text")
                    track_state["face"] = attrs.get("face_name")
                    track_state["last_logged"] = now_ts
                elif time_since_log >= 60.0:
                    should_log = True
                    track_state["last_logged"] = now_ts

        # Cleanup old throttle state if cache grows large
        if len(_ENTITY_LOG_THROTTLE) > 200:
            stale = [k for k, v in _ENTITY_LOG_THROTTLE.items() if (now_ts - v.get("last_logged", 0)) > 300.0]
            for k in stale:
                _ENTITY_LOG_THROTTLE.pop(k, None)

        if should_log:
            ent_id = str(uuid.uuid4())
            thumb_rel = None

            # For alerts or notable entities, save a thumbnail crop
            if is_alert:
                thumb_name = f"thumb_{ent_id[:8]}.webp"
                thumb_file = thumb_dir / thumb_name
                try:
                    x1, y1, x2, y2 = bbox
                    pad = 12
                    crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                    if crop.size > 0:
                        cv2.imwrite(str(thumb_file), crop, [cv2.IMWRITE_WEBP_QUALITY, 80])
                        thumb_rel = f"/storage/thumbnails/{thumb_name}"
                except Exception as e:
                    logger.warning(f"Error saving alert thumbnail: {e}")

            clip_path = camera.stream_url if (camera and camera.stream_url) else None

            log_entry = EntityLog(
                id=ent_id,
                camera_id=camera_id,
                timestamp=datetime.now(timezone.utc),
                entity_type=ent_type,
                upper_color=attrs.get("upper_color"),
                lower_color=attrs.get("lower_color"),
                height_cm=attrs.get("height_cm"),
                gender=attrs.get("gender"),
                plate_text=attrs.get("plate_text"),
                vehicle_type=attrs.get("vehicle_type"),
                direction=attrs.get("direction"),
                speed_kmh=attrs.get("speed_kmh"),
                face_name=attrs.get("face_name"),
                skin_tone=attrs.get("skin_tone"),
                location_lat=ent_lat,
                location_lon=ent_lon,
                trajectory_id=str(tid_val) if tid_val is not None else None,
                is_alert=is_alert,
                alert_type=alert_type,
                confidence_score=conf_val,
                clip_path=clip_path,
                thumbnail_path=thumb_rel,
                retention_tier="protected" if is_alert else "passive",
                rule_fired=rule_fired,
                acknowledged=False,
                is_low_light=bool(attrs.get("is_low_light", False)),
                posture=posture
            )
            try:
                db.add(log_entry)
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning(f"Failed to commit entity log: {e}")

        # Add threat annotations to attributes for UI rendering
        attrs["is_alert"] = is_alert
        attrs["alert_type"] = alert_type
        attrs["rule_fired"] = rule_fired
        attrs["threat_score"] = threat_score

        detected_out.append(DetectedEntityOut(
            entity_type=ent_type,
            bbox=bbox,
            confidence=conf_val,
            track_id=tid_val,
            attributes=attrs
        ))

    return FrameDetectResponse(
        camera_id=camera_id,
        entities=detected_out,
        frame_width=w,
        frame_height=h
    )
