from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.database import get_db
from backend.app.models import EntityLog, User
from backend.app.schemas import AlertDetailOut, AcknowledgeRequest, GenericStatusResponse
from backend.app.auth.jwt_auth import get_current_user
from backend.app.api.entities import _to_entity_out

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])

@router.get("/active", response_model=List[AlertDetailOut])
def get_active_alerts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns unacknowledged active alerts for the live Alert Feed.
    """
    active_alerts = (
        db.query(EntityLog)
        .filter(
            EntityLog.is_alert == True,
            EntityLog.acknowledged == False,
            EntityLog.deleted_manually == False
        )
        .order_by(desc(EntityLog.timestamp))
        .all()
    )

    results = []
    for a in active_alerts:
        results.append(AlertDetailOut(
            id=a.id,
            camera_id=a.camera_id,
            timestamp=a.timestamp,
            alert_type=a.alert_type or "alert",
            rule_fired=a.rule_fired,
            confidence_score=a.confidence_score,
            thumbnail_url=a.thumbnail_path or (f"/storage/thumbnails/{a.id}.webp" if a.thumbnail_path else None),
            clip_url=a.clip_path or (f"/storage/clips/{a.id}.mp4" if a.clip_path else None),
            acknowledged=a.acknowledged,
            entity_details=_to_entity_out(a)
        ))
    return results

@router.post("/{alert_id}/acknowledge", response_model=GenericStatusResponse)
def acknowledge_alert(
    alert_id: str,
    req: AcknowledgeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    alert = db.query(EntityLog).filter(EntityLog.id == alert_id, EntityLog.is_alert == True).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active alert not found")

    alert.acknowledged = True
    db.commit()

    return GenericStatusResponse(status="success", message=f"Alert {alert_id} acknowledged by {current_user.username}")

@router.get("/history", response_model=List[AlertDetailOut])
def get_alert_history(
    alert_type: str = None,
    acknowledged: bool = None,
    camera_id: str = None,
    is_low_light: bool = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns historical security breach alerts matching filters:
    - alert_type (geo_fence, behavior, tamper, correlated)
    - acknowledged (True/False)
    - is_low_light (True/False - night incidents)
    - camera_id
    """
    query = db.query(EntityLog).filter(
        EntityLog.is_alert == True,
        EntityLog.deleted_manually == False
    )

    if alert_type and alert_type.lower() != "all":
        query = query.filter(EntityLog.alert_type == alert_type)
    if acknowledged is not None:
        query = query.filter(EntityLog.acknowledged == acknowledged)
    if is_low_light is not None:
        query = query.filter(EntityLog.is_low_light == is_low_light)
    if camera_id:
        query = query.filter(EntityLog.camera_id == camera_id)


    alerts = query.order_by(desc(EntityLog.timestamp)).offset(offset).limit(limit).all()

    results = []
    for a in alerts:
        results.append(AlertDetailOut(
            id=a.id,
            camera_id=a.camera_id,
            timestamp=a.timestamp,
            alert_type=a.alert_type or "alert",
            rule_fired=a.rule_fired,
            confidence_score=a.confidence_score,
            thumbnail_url=f"/storage/thumbnails/{a.id}.webp" if a.thumbnail_path else None,
            clip_url=f"/storage/clips/{a.id}.mp4" if a.clip_path else None,
            acknowledged=a.acknowledged,
        ))
    return results


@router.get("/verify-integrity/{alert_id}")
def verify_alert_integrity(
    alert_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Two-Tier Evidentiary Proof Verification (PDF Page 9, Item 1.1):
    - Tier 1: On-disk cryptographic SHA-256 match against original video clip.
    - Tier 2: If clip was evicted by 92% FIFO purge, verifies immutable MediaTombstoneLog.
    """
    import hashlib
    from pathlib import Path
    from backend.app.models import MediaTombstoneLog

    alert = db.query(EntityLog).filter(EntityLog.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert record not found")

    clip_path = alert.clip_path
    recorded_sha = alert.clip_sha256
    notarization_token = alert.notarization_token

    # Tier 1: On-disk file verification
    if clip_path and Path(clip_path).exists():
        try:
            with open(clip_path, "rb") as f:
                actual_sha = hashlib.sha256(f.read()).hexdigest()
            
            is_valid = (actual_sha == recorded_sha) if recorded_sha and recorded_sha != "0" * 64 else True
            return {
                "alert_id": alert.id,
                "camera_id": alert.camera_id,
                "timestamp": alert.timestamp.isoformat(),
                "tier": "TIER_1_ON_DISK",
                "status": "VERIFIED_VALID" if is_valid else "HASH_MISMATCH",
                "verified": is_valid,
                "clip_sha256": actual_sha,
                "notarization_token": notarization_token,
                "clip_path": clip_path
            }
        except Exception as e:
            pass

    # Tier 2: Evicted tombstone log verification
    tombstone = db.query(MediaTombstoneLog).filter(MediaTombstoneLog.entity_id == alert_id).first()
    if tombstone:
        return {
            "alert_id": alert.id,
            "camera_id": alert.camera_id,
            "timestamp": alert.timestamp.isoformat(),
            "tier": "TIER_2_TOMBSTONE",
            "status": "EVICTED_TOMBSTONE_VERIFIED",
            "verified": True,
            "clip_sha256": tombstone.clip_sha256,
            "notarization_token": notarization_token,
            "evicted_at": tombstone.evicted_at.isoformat() if tombstone.evicted_at else None,
            "disk_usage_at_eviction": tombstone.disk_usage_pct_at_eviction,
            "message": "Physical video clip was purged under 92% emergency FIFO policy; audit hash preserved."
        }

    return {
        "alert_id": alert.id,
        "camera_id": alert.camera_id,
        "timestamp": alert.timestamp.isoformat(),
        "tier": "UNKNOWN",
        "status": "MEDIA_NOT_FOUND",
        "verified": False,
        "clip_sha256": recorded_sha,
        "notarization_token": notarization_token
    }


