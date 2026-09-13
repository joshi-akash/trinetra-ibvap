import uuid
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_

from backend.app.database import get_db
from backend.app.models import EntityLog, CameraRegistry, FalseFlagLog, AuditLog, User
from backend.app.schemas import (
    EntityLogOut, FalseFlagRequest, FalseFlagOut,
    ExportRequest, ExportResponse, GenericStatusResponse,
    FrameAnalysis
)
from backend.app.auth.jwt_auth import get_current_user, require_roles
from backend.app.ingestion.rtsp_service import rtsp_service
from backend.app.rule_engine.bifurcation import bifurcation_engine
from backend.app.alert_manager.package_builder import alert_manager
from backend.app.export_service.export_builder import export_service

router = APIRouter(prefix="/api/entities", tags=["Entities"])

def _to_entity_out(e: EntityLog) -> EntityLogOut:
    return EntityLogOut(
        id=e.id,
        camera_id=e.camera_id,
        timestamp=e.timestamp,
        entity_type=e.entity_type,
        upper_color=e.upper_color,
        lower_color=e.lower_color,
        height_cm=e.height_cm,
        gender=e.gender,
        plate_text=e.plate_text,
        vehicle_type=getattr(e, "vehicle_type", None),
        direction=getattr(e, "direction", None),
        speed_kmh=getattr(e, "speed_kmh", None),
        face_name=getattr(e, "face_name", None),
        skin_tone=getattr(e, "skin_tone", None),
        location={"lat": e.location_lat, "lon": e.location_lon} if e.location_lat and e.location_lon else None,
        trajectory_id=e.trajectory_id,
        is_alert=e.is_alert,
        alert_type=e.alert_type,
        confidence_score=e.confidence_score,
        clip_path=e.clip_path,
        thumbnail_path=e.thumbnail_path,
        retention_tier=e.retention_tier,
        deleted_manually=e.deleted_manually,
        rule_fired=e.rule_fired,
        acknowledged=e.acknowledged,
        is_low_light=getattr(e, "is_low_light", False),
        posture=getattr(e, "posture", None)
    )

@router.get("", response_model=List[EntityLogOut])
def list_entities(
    camera_id: Optional[str] = None,
    entity_type: Optional[str] = None,
    is_alert: Optional[bool] = None,
    color: Optional[str] = None,
    posture: Optional[str] = None,
    is_low_light: Optional[bool] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(EntityLog).filter(EntityLog.deleted_manually == False)

    if camera_id:
        query = query.filter(EntityLog.camera_id == camera_id)
    if entity_type:
        query = query.filter(EntityLog.entity_type == entity_type)
    if is_alert is not None:
        query = query.filter(EntityLog.is_alert == is_alert)
    if is_low_light is not None:
        query = query.filter(EntityLog.is_low_light == is_low_light)
    if posture:
        query = query.filter(EntityLog.posture == posture)
    if color:
        query = query.filter(
            or_(
                EntityLog.upper_color.ilike(f"%{color}%"),
                EntityLog.lower_color.ilike(f"%{color}%")
            )
        )
    if start_time:
        query = query.filter(EntityLog.timestamp >= start_time)
    if end_time:
        query = query.filter(EntityLog.timestamp <= end_time)

    records = query.order_by(desc(EntityLog.timestamp)).offset(offset).limit(limit).all()
    return [_to_entity_out(r) for r in records]

@router.post("/ingest", response_model=GenericStatusResponse)
def ingest_frame_analysis(
    frame: FrameAnalysis,
    db: Session = Depends(get_db)
):
    """
    In-process / HTTP ingestion hook for FrameAnalysis from AI Pipeline (Person A/B).
    Passes frame through the Rule Engine (bifurcation), saves to entity_log,
    and builds alert package if an active alert fires.
    Fully darkness-aware (preserves frame_quality.low_light).
    """
    # 1. Push frame reference to camera ring buffer
    rtsp_service.push_frame(
        camera_id=frame.camera_id,
        frame_id=frame.frame_ref,
        timestamp=frame.timestamp
    )

    # 2. Fetch camera geo-fence polygon
    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == frame.camera_id).first()
    geofence_coords = camera.geo_fence_polygon if camera else None

    # 3. Bifurcation Evaluation
    decisions, camera_alert = bifurcation_engine.evaluate_frame(frame, geofence_coords=geofence_coords)

    alerts_created = 0
    passive_created = 0
    is_low_light = bool(frame.frame_quality.low_light)

    # 4. Handle Camera Level Tamper Alert
    if camera_alert:
        alert_manager.build_and_save_alert(
            db=db,
            camera_id=frame.camera_id,
            timestamp=frame.timestamp,
            entity_detection=None,
            alert_type=camera_alert["alert_type"],
            rule_fired=camera_alert["rule_fired"],
            confidence_score=camera_alert["confidence_score"],
            is_low_light=is_low_light
        )
        if camera:
            camera.trust_score = max(0.1, camera.trust_score - 0.2)
            camera.last_tamper_check = frame.timestamp
            db.commit()
        alerts_created += 1

    # 5. Handle Each Entity Decision
    for decision in decisions:
        ent = decision.entity_detection
        detected_posture = ent.attributes.posture
        if decision.is_alert:
            # Active Alert Path: Assemble JSON + WebP thumbnail + 30s clip (FR-ALR-03)
            alert_manager.build_and_save_alert(
                db=db,
                camera_id=frame.camera_id,
                timestamp=frame.timestamp,
                entity_detection=ent,
                alert_type=decision.alert_type,
                rule_fired=decision.rule_fired,
                confidence_score=decision.confidence_score,
                frame_ref=frame.frame_ref,
                is_low_light=is_low_light,
                posture=detected_posture
            )
            alerts_created += 1
        else:
            # Passive Logging Path: Log quietly, no siren, no clip (FR-ALR-01)
            passive_record = EntityLog(
                id=str(uuid.uuid4()),
                camera_id=frame.camera_id,
                timestamp=frame.timestamp,
                entity_type=ent.entity_type,
                upper_color=ent.attributes.upper_color,
                lower_color=ent.attributes.lower_color,
                height_cm=ent.attributes.height_cm,
                gender=ent.attributes.gender,
                plate_text=ent.attributes.plate_text,
                vehicle_type=getattr(ent.attributes, "vehicle_type", None),
                direction=getattr(ent.attributes, "direction", None),
                speed_kmh=getattr(ent.attributes, "speed_kmh", None),
                face_name=getattr(ent.attributes, "face_name", None) or (ent.attributes.face_match.suspect_id if ent.attributes.face_match else None),
                skin_tone=getattr(ent.attributes, "skin_tone", None),
                location_lat=ent.location.lat,
                location_lon=ent.location.lon,
                trajectory_id=ent.track_id,
                is_alert=False,
                alert_type=None,
                confidence_score=decision.confidence_score,
                retention_tier="passive",
                deleted_manually=False,
                rule_fired=None,
                is_low_light=is_low_light,
                posture=detected_posture
            )
            db.add(passive_record)
            passive_created += 1

    db.commit()
    return GenericStatusResponse(
        status="success",
        message=f"Processed frame for {frame.camera_id}: {alerts_created} active alerts, {passive_created} passive logs."
    )


@router.post("/{entity_id}/false-flag", response_model=FalseFlagOut)
def mark_false_flag(
    entity_id: str,
    req: FalseFlagRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    entity = db.query(EntityLog).filter(EntityLog.id == entity_id).first()
    if not entity:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entity record not found")

    flag_id = str(uuid.uuid4())
    false_flag = FalseFlagLog(
        id=flag_id,
        entity_log_id=entity.id,
        marked_by=current_user.username,
        marked_at=datetime.now(timezone.utc),
        moved_to_hard_negatives=req.moved_to_hard_negatives,
        notes=req.notes
    )
    db.add(false_flag)

    # Update camera trust score downward slightly due to false positive
    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == entity.camera_id).first()
    if camera and camera.trust_score > 0.3:
        camera.trust_score = round(max(0.1, camera.trust_score - 0.02), 2)

    db.commit()
    return FalseFlagOut(
        id=false_flag.id,
        entity_log_id=false_flag.entity_log_id,
        marked_by=false_flag.marked_by,
        marked_at=false_flag.marked_at,
        moved_to_hard_negatives=false_flag.moved_to_hard_negatives,
        notes=false_flag.notes
    )

@router.post("/{entity_id}/export", response_model=ExportResponse)
def export_entity_record(
    entity_id: str,
    req: ExportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["commander", "admin"]))  # FR-EXP-03: Operators cannot export
):
    try:
        export_result = export_service.build_export_package(
            db=db,
            entity_id=entity_id,
            exporting_user=current_user,
            include_clip=req.include_clip,
            channel=req.channel
        )
        return ExportResponse(**export_result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.delete("/{entity_id}", response_model=GenericStatusResponse)
def manually_delete_entity(
    entity_id: str,
    reason: str = Query(..., min_length=3, description="Mandatory audit reason for deletion"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["commander", "admin"]))
):
    """
    FR-RET-02: Deletion of protected records is never automatic.
    Always requires explicit human action with a recorded reason in deletion_audit and audit_log.
    """
    entity = db.query(EntityLog).filter(EntityLog.id == entity_id).first()
    if not entity:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entity record not found")

    deletion_audit_data = {
        "deleted_by_user_id": current_user.id,
        "deleted_by_username": current_user.username,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "previous_retention_tier": entity.retention_tier
    }

    entity.deleted_manually = True
    entity.deletion_audit = deletion_audit_data

    # Log to system audit_log
    audit = AuditLog(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        action="MANUAL_RECORD_DELETION",
        target_id=entity.id,
        timestamp=datetime.now(timezone.utc),
        details=deletion_audit_data
    )
    db.add(audit)
    db.commit()

    return GenericStatusResponse(
        status="success",
        message=f"Entity {entity_id} marked deleted with audit log recorded."
    )
