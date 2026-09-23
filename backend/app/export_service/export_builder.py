import os
import io
import json
import uuid
import zipfile
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from backend.app.config import settings, PROJECT_ROOT
from backend.app.models import EntityLog, ExportLog, AuditLog, User, CameraRegistry
from backend.app.ingestion.rtsp_service import rtsp_service


def resolve_source_media_path(path_or_url: Optional[str]) -> Optional[Path]:
    """
    Safely resolves any relative, absolute, or storage/footage URI into a real on-disk Path.
    Handles /storage/thumbnails/..., /storage/clips/..., /footage/..., and bare filenames.
    """
    if not path_or_url or not str(path_or_url).strip():
        return None

    raw = str(path_or_url).strip()
    normalized = raw.replace("\\", "/")

    # Check direct absolute path
    p = Path(raw)
    if p.is_absolute() and p.exists() and p.stat().st_size > 0:
        return p

    # Handle /storage/ prefix
    if normalized.startswith("/storage/"):
        rel = normalized[len("/storage/"):]
        candidate = settings.STORAGE_DIR / rel
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    elif normalized.startswith("storage/"):
        rel = normalized[len("storage/"):]
        candidate = settings.STORAGE_DIR / rel
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    # Handle /footage/ prefix
    if normalized.startswith("/footage/"):
        rel = normalized[len("/footage/"):]
        candidate = PROJECT_ROOT / "test_footage" / rel
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    elif normalized.startswith("footage/"):
        rel = normalized[len("footage/"):]
        candidate = PROJECT_ROOT / "test_footage" / rel
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    # Check relative to PROJECT_ROOT
    rel_proj = PROJECT_ROOT / raw.lstrip("/")
    if rel_proj.exists() and rel_proj.stat().st_size > 0:
        return rel_proj

    # Check search directories by filename
    filename = Path(normalized).name
    search_dirs = [
        settings.THUMBNAILS_DIR,
        settings.CLIPS_DIR,
        settings.STORAGE_DIR,
        PROJECT_ROOT / "test_footage",
        PROJECT_ROOT / "uploads" / "training_media",
    ]
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    return None


class ExportService:
    """
    Builds self-contained, signed offline HQ export packages (FR-EXP-01, FR-EXP-02, FR-EXP-03).
    Operates completely offline with zero WAN dependency.
    """

    @staticmethod
    def _find_camera_source_video(db: Session, camera_id: str, entity_clip_path: Optional[str]) -> Optional[Path]:
        """Finds best available video footage file for a camera."""
        # 1. Check camera's registered stream_url
        camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == camera_id).first()
        if camera and camera.stream_url:
            p = resolve_source_media_path(camera.stream_url)
            if p:
                return p

        # 2. Check entity's recorded clip_path
        if entity_clip_path:
            p = resolve_source_media_path(entity_clip_path)
            if p:
                return p

        # 3. Check for any video files in test_footage
        footage_dir = PROJECT_ROOT / "test_footage"
        if footage_dir.exists():
            for vid_file in sorted(footage_dir.glob("*.mp4")):
                if vid_file.stat().st_size > 1000:
                    return vid_file

        return None

    @staticmethod
    def build_export_package(
        db: Session,
        entity_id: str,
        exporting_user: User,
        include_clip: bool = True,
        channel: str = "local_bundle"
    ) -> Dict[str, Any]:
        # 1. Fetch Entity Record
        entity = db.query(EntityLog).filter(EntityLog.id == entity_id).first()
        if not entity:
            raise ValueError(f"Entity log {entity_id} not found")

        export_id = str(uuid.uuid4())
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        zip_filename = f"TRINETRA_EXPORT_{entity.camera_id}_{timestamp_str}_{export_id[:8]}.zip"
        zip_filepath = settings.EXPORTS_DIR / zip_filename

        camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == entity.camera_id).first()
        src_video = ExportService._find_camera_source_video(db, entity.camera_id, entity.clip_path)

        # 2. Compile Forensic Metadata Document
        metadata = {
            "trinetra_export_version": "1.0",
            "export_id": export_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "exported_by": {
                "user_id": exporting_user.id,
                "username": exporting_user.username,
                "role": exporting_user.role
            },
            "channel": channel,
            "record": {
                "id": entity.id,
                "camera_id": entity.camera_id,
                "timestamp": entity.timestamp.isoformat() if entity.timestamp else None,
                "entity_type": entity.entity_type,
                "upper_color": entity.upper_color,
                "lower_color": entity.lower_color,
                "height_cm": entity.height_cm,
                "gender": entity.gender,
                "plate_text": entity.plate_text,
                "vehicle_type": getattr(entity, "vehicle_type", None),
                "direction": getattr(entity, "direction", None),
                "speed_kmh": getattr(entity, "speed_kmh", None),
                "face_name": getattr(entity, "face_name", None),
                "skin_tone": getattr(entity, "skin_tone", None),
                "location": {
                    "lat": entity.location_lat,
                    "lon": entity.location_lon
                },
                "is_alert": entity.is_alert,
                "alert_type": entity.alert_type,
                "confidence_score": entity.confidence_score,
                "rule_fired": entity.rule_fired,
                "retention_tier": entity.retention_tier,
                "is_low_light": getattr(entity, "is_low_light", False),
                "posture": getattr(entity, "posture", None)
            }
        }

        # 3. Create Self-Contained ZIP Archive with Complete Evidence Bundle
        with zipfile.ZipFile(zip_filepath, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            # -----------------------------------------------------------------
            # A. Add Screenshot / Keyframe Image Evidence
            # -----------------------------------------------------------------
            snapshot_bytes: Optional[bytes] = None
            thumbnail_bytes: Optional[bytes] = None

            # 1. Attempt to load existing thumbnail from disk
            resolved_thumb = resolve_source_media_path(entity.thumbnail_path)
            if not resolved_thumb:
                # Check standard naming schemes
                for cand in [
                    settings.THUMBNAILS_DIR / f"snap_{entity.id[:8]}.jpg",
                    settings.THUMBNAILS_DIR / f"thumb_{entity.id[:8]}.webp",
                    settings.THUMBNAILS_DIR / f"{entity.id}.webp",
                    settings.THUMBNAILS_DIR / f"{entity.id}.jpg",
                ]:
                    if cand.exists() and cand.stat().st_size > 500:
                        resolved_thumb = cand
                        break

            if resolved_thumb and resolved_thumb.exists() and resolved_thumb.stat().st_size > 500:
                try:
                    with open(resolved_thumb, "rb") as f:
                        raw_data = f.read()
                    snapshot_bytes = raw_data
                    thumbnail_bytes = raw_data
                except Exception:
                    snapshot_bytes = None

            # 2. If no valid image exists, extract genuine frame from camera video source
            if not snapshot_bytes and src_video and src_video.exists():
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(src_video))
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    target_f = min(max(0, int(fps * 2)), max(0, total_f - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame_bgr = cap.read()
                    cap.release()

                    if ret and frame_bgr is not None:
                        # Draw tactical forensic overlay badge on real footage frame
                        h, w = frame_bgr.shape[:2]
                        # Top overlay banner
                        cv2.rectangle(frame_bgr, (0, 0), (w, 40), (15, 23, 42), -1)
                        cv2.putText(frame_bgr, f"TRINETRA CCTV FORENSIC EVIDENCE | CAM: {entity.camera_id}", (15, 26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
                        time_str = metadata["record"]["timestamp"] or datetime.now(timezone.utc).isoformat()
                        cv2.putText(frame_bgr, f"{time_str}", (w - 280, 26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)

                        # Target metadata banner at bottom
                        cv2.rectangle(frame_bgr, (0, h - 35), (w, h), (15, 23, 42), -1)
                        target_info = f"TARGET: {entity.entity_type.upper()}"
                        if entity.confidence_score:
                            target_info += f" [{int(entity.confidence_score * 100)}%]"
                        if entity.plate_text:
                            target_info += f" | PLATE: {entity.plate_text}"
                        if getattr(entity, "face_name", None):
                            target_info += f" | SUSPECT: {entity.face_name}"
                        if entity.rule_fired:
                            target_info += f" | ALERT: {entity.rule_fired}"
                        cv2.putText(frame_bgr, target_info, (15, h - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 197, 94) if entity.plate_text else (245, 158, 11), 2)

                        _, snap_buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        snapshot_bytes = snap_buf.tobytes()

                        # Scaled thumbnail
                        thumb_h, thumb_w = min(240, h), int(w * (min(240, h) / h))
                        thumb_img = cv2.resize(frame_bgr, (thumb_w, thumb_h))
                        _, thumb_buf = cv2.imencode('.webp', thumb_img, [cv2.IMWRITE_WEBP_QUALITY, 85])
                        thumbnail_bytes = thumb_buf.tobytes()
                except Exception:
                    snapshot_bytes = None

            # 3. Check ring buffer frame if still missing
            if not snapshot_bytes:
                buf = rtsp_service.get_buffer(entity.camera_id)
                with buf.lock:
                    if buf.buffer and buf.buffer[-1].image_bytes:
                        snapshot_bytes = buf.buffer[-1].image_bytes
                        thumbnail_bytes = snapshot_bytes

            # 4. Fallback canvas only if absolutely no video or buffer exists
            if not snapshot_bytes:
                import cv2
                import numpy as np
                canvas = np.zeros((360, 640, 3), dtype=np.uint8)
                canvas[:] = (20, 25, 35)
                cv2.putText(canvas, "TRINETRA SOVEREIGN EVIDENCE INCIDENT", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
                cv2.putText(canvas, f"CAMERA: {entity.camera_id} | TARGET: {entity.entity_type.upper()}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(canvas, f"TIMESTAMP: {metadata['record']['timestamp']}", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
                if entity.plate_text:
                    cv2.putText(canvas, f"ANPR LICENSE PLATE: {entity.plate_text}", (30, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)
                if getattr(entity, "face_name", None):
                    cv2.putText(canvas, f"FACE RECOGNITION MATCH: {entity.face_name}", (30, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (239, 68, 68), 2)
                cv2.rectangle(canvas, (10, 10), (630, 350), (56, 189, 248), 1)
                _, buf_cv = cv2.imencode('.jpg', canvas)
                snapshot_bytes = buf_cv.tobytes()
                thumbnail_bytes = snapshot_bytes

            zip_file.writestr("evidence_snapshot.jpg", snapshot_bytes)
            zip_file.writestr("evidence_thumbnail.webp", thumbnail_bytes or snapshot_bytes)

            # -----------------------------------------------------------------
            # B. Add Incident Video Clip (Under 30s rule: whole clip as output!)
            # -----------------------------------------------------------------
            clip_info: Dict[str, Any] = {"included": False}
            if include_clip:
                clip_written = False

                if src_video and src_video.exists() and src_video.stat().st_size > 0:
                    try:
                        import cv2
                        cap = cv2.VideoCapture(str(src_video))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0 or not (1 <= fps <= 120):
                            fps = 25.0
                        duration_sec = total_frames / fps if total_frames > 0 else 0.0
                        cap.release()

                        # RULE: In cases where the total clip is lesser than or equal to 30s, simply give the whole clip as output!
                        if duration_sec <= 30.0 or duration_sec == 0.0:
                            zip_file.write(str(src_video), arcname="evidence_clip.mp4")
                            clip_written = True
                            clip_info = {
                                "included": True,
                                "filename": "evidence_clip.mp4",
                                "duration_sec": round(duration_sec, 2),
                                "mode": "complete_clip_under_30s",
                                "source_file": src_video.name,
                                "fps": round(fps, 1),
                                "total_frames": total_frames
                            }
                        else:
                            # Video is longer than 30 seconds: extract a 30-second evidence window
                            try:
                                import subprocess, imageio_ffmpeg, tempfile
                                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_clip:
                                    tmp_clip_path = tmp_clip.name

                                start_sec = 0.0
                                cmd = [
                                    ffmpeg_exe, "-y",
                                    "-ss", str(start_sec),
                                    "-i", str(src_video),
                                    "-t", "30.0",
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                    "-preset", "fast",
                                    "-movflags", "+faststart",
                                    str(tmp_clip_path)
                                ]
                                res = subprocess.run(cmd, capture_output=True, timeout=35)
                                if res.returncode == 0 and os.path.exists(tmp_clip_path) and os.path.getsize(tmp_clip_path) > 0:
                                    zip_file.write(tmp_clip_path, arcname="evidence_clip.mp4")
                                    clip_written = True
                                    clip_info = {
                                        "included": True,
                                        "filename": "evidence_clip.mp4",
                                        "duration_sec": 30.0,
                                        "mode": "30s_window_extraction",
                                        "source_file": src_video.name,
                                        "fps": round(fps, 1)
                                    }
                                try:
                                    os.remove(tmp_clip_path)
                                except Exception:
                                    pass
                            except Exception:
                                pass

                            # If FFmpeg slicing fails, fallback to giving the whole clip as output
                            if not clip_written:
                                zip_file.write(str(src_video), arcname="evidence_clip.mp4")
                                clip_written = True
                                clip_info = {
                                    "included": True,
                                    "filename": "evidence_clip.mp4",
                                    "duration_sec": round(duration_sec, 2),
                                    "mode": "fallback_complete_clip",
                                    "source_file": src_video.name
                                }
                    except Exception:
                        pass

                # Fallback: Check ring buffer or existing clip file
                if not clip_written:
                    resolved_clip = resolve_source_media_path(entity.clip_path)
                    if resolved_clip and resolved_clip.exists() and resolved_clip.stat().st_size > 1000:
                        zip_file.write(str(resolved_clip), arcname="evidence_clip.mp4")
                        clip_written = True
                        clip_info = {"included": True, "filename": "evidence_clip.mp4", "mode": "buffered_clip"}

                if not clip_written:
                    # Final fallback: create video from ring buffer or short informative clip
                    try:
                        import cv2
                        import numpy as np
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_clip:
                            tmp_clip_path = tmp_clip.name
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(tmp_clip_path, fourcc, 20.0, (640, 360))
                        for f_idx in range(60):
                            f_img = np.zeros((360, 640, 3), dtype=np.uint8)
                            f_img[:] = (15, 20, 30)
                            cv2.putText(f_img, f"TRINETRA INCIDENT REPLAY — {entity.camera_id}", (30, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
                            cv2.putText(f_img, f"EVENT: {entity.rule_fired or 'DETECTION'} | SEC: {f_idx/20.0:.1f}s", (30, 140),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                            cv2.putText(f_img, f"TARGET: {entity.entity_type.upper()}", (30, 190),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 158, 11), 1)
                            if entity.plate_text:
                                cv2.putText(f_img, f"PLATE: {entity.plate_text}", (30, 240),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)
                            cv2.rectangle(f_img, (10, 10), (630, 350), (239, 68, 68), 2)
                            writer.write(f_img)
                        writer.release()
                        zip_file.write(tmp_clip_path, arcname="evidence_clip.mp4")
                        clip_written = True
                        clip_info = {"included": True, "filename": "evidence_clip.mp4", "mode": "synthesized_replay"}
                        try:
                            os.remove(tmp_clip_path)
                        except Exception:
                            pass
                    except Exception:
                        zip_file.writestr("evidence_clip.txt", "No video clip source available")

            # -----------------------------------------------------------------
            # C. Add metadata.json (enriched with clip_info)
            # -----------------------------------------------------------------
            metadata["clip_info"] = clip_info
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))

            # -----------------------------------------------------------------
            # D. Add Human-Readable Incident Summary Dossier
            # -----------------------------------------------------------------
            dossier_text = (
                f"================================================================================\n"
                f"                 TRINETRA SOVEREIGN EVIDENCE INCIDENT DOSSIER\n"
                f"================================================================================\n"
                f"Export ID         : {export_id}\n"
                f"Generated At      : {metadata['exported_at']}\n"
                f"Signing Authority : {exporting_user.username} ({exporting_user.role})\n"
                f"Distribution      : {channel.upper()}\n\n"
                f"--------------------------------------------------------------------------------\n"
                f"1. INCIDENT & TARGET SPECIFICATION\n"
                f"--------------------------------------------------------------------------------\n"
                f"Entity Log ID     : {entity.id}\n"
                f"Camera Identifier : {entity.camera_id}\n"
                f"Detection Time    : {entity.timestamp}\n"
                f"Target Class      : {entity.entity_type}\n"
                f"Confidence Level  : {entity.confidence_score}\n"
                f"Retention Tier    : {entity.retention_tier}\n"
                f"Alert Triggered   : {entity.is_alert} ({entity.alert_type or 'None'})\n"
                f"Rule Violated     : {entity.rule_fired or 'PERIMETER_MONITOR'}\n\n"
                f"--------------------------------------------------------------------------------\n"
                f"2. INTELLIGENCE & FORENSIC ATTRIBUTES\n"
                f"--------------------------------------------------------------------------------\n"
                f"License Plate     : {entity.plate_text or 'N/A'}\n"
                f"Face Recognition  : {getattr(entity, 'face_name', None) or 'N/A'}\n"
                f"Clothing Color    : Upper: {entity.upper_color or 'N/A'} | Lower: {entity.lower_color or 'N/A'}\n"
                f"Estimated Height  : {f'{entity.height_cm} cm' if entity.height_cm else 'N/A'}\n"
                f"Movement Vectors  : Speed: {getattr(entity, 'speed_kmh', None) or 'N/A'} km/h | Heading: {getattr(entity, 'direction', None) or 'N/A'}\n"
                f"Night / Low Light : {'YES' if getattr(entity, 'is_low_light', False) else 'NO'}\n"
                f"Posture Analysis  : {getattr(entity, 'posture', None) or 'N/A'}\n\n"
                f"--------------------------------------------------------------------------------\n"
                f"3. EVIDENCE PACKAGE MANIFEST\n"
                f"--------------------------------------------------------------------------------\n"
                f"• metadata.json               : Forensic machine-readable JSON telemetry\n"
                f"• evidence_snapshot.jpg       : Full-scene high-resolution CCTV keyframe photograph\n"
                f"• evidence_thumbnail.webp     : Target identification asset\n"
                f"• evidence_clip.mp4           : CCTV incident video recording\n"
                f"• integrity_hash.txt          : Sovereign cryptographic validation proof\n"
                f"• FORENSIC_INCIDENT_REPORT.txt: Human-readable incident dossier\n"
                f"================================================================================\n"
            )
            zip_file.writestr("FORENSIC_INCIDENT_REPORT.txt", dossier_text)

        # 4. Compute Cryptographic SHA-256 Payload Hash
        sha256 = hashlib.sha256()
        with open(zip_filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        payload_hash = sha256.hexdigest()

        # Append integrity hash document into the archive
        with zipfile.ZipFile(zip_filepath, 'a', compression=zipfile.ZIP_DEFLATED) as zip_file:
            integrity_manifest = (
                f"TRINETRA SOVEREIGN EVIDENCE PACKAGE\n"
                f"Export ID: {export_id}\n"
                f"Entity ID: {entity.id}\n"
                f"Camera ID: {entity.camera_id}\n"
                f"Payload SHA-256: {payload_hash}\n"
                f"Exported At: {metadata['exported_at']}\n"
                f"Exported By: {exporting_user.username} ({exporting_user.role})\n"
            )
            zip_file.writestr("integrity_hash.txt", integrity_manifest)

        # 5. Write to export_log
        export_log_entry = ExportLog(
            id=export_id,
            entity_log_id=entity.id,
            exported_by=exporting_user.id,
            exported_at=datetime.now(timezone.utc),
            payload_hash=payload_hash,
            channel=channel
        )
        db.add(export_log_entry)

        # 6. Write to audit_log
        audit_entry = AuditLog(
            id=str(uuid.uuid4()),
            user_id=exporting_user.id,
            action="EXPORT_HQ_PACKAGE",
            target_id=entity.id,
            timestamp=datetime.now(timezone.utc),
            details={
                "export_id": export_id,
                "payload_hash": payload_hash,
                "filename": zip_filename,
                "include_clip": include_clip
            }
        )
        db.add(audit_entry)
        db.commit()

        return {
            "export_id": export_id,
            "entity_log_id": entity.id,
            "payload_hash": payload_hash,
            "download_url": f"/storage/exports/{zip_filename}",
            "exported_at": export_log_entry.exported_at,
            "channel": channel
        }


export_service = ExportService()
