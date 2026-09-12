import os
import io
import json
import uuid
import zipfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models import EntityLog, ExportLog, AuditLog, User

class ExportService:
    """
    Builds self-contained, signed offline HQ export packages (FR-EXP-01, FR-EXP-02, FR-EXP-03).
    Operates completely offline with zero WAN dependency.
    """

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

        # 2. Compile Metadata Document
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

        # 3. Create Self-Contained ZIP Archive
        with zipfile.ZipFile(zip_filepath, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            # Add metadata.json
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))

            # Add thumbnail if present
            if entity.thumbnail_path and os.path.exists(entity.thumbnail_path):
                zip_file.write(entity.thumbnail_path, arcname="evidence_thumbnail.webp")
            else:
                # Add synthetic thumbnail marker
                zip_file.writestr("evidence_thumbnail.txt", "Thumbnail not attached or degraded")

            # Add clip if requested and present
            if include_clip and entity.clip_path and os.path.exists(entity.clip_path):
                zip_file.write(entity.clip_path, arcname="evidence_30s_clip.mp4")

        # 4. Compute Cryptographic SHA-256 Payload Hash
        sha256 = hashlib.sha256()
        with open(zip_filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        payload_hash = sha256.hexdigest()

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
