import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, JSON, Index, Integer
from sqlalchemy.orm import relationship
from backend.app.database import Base

def utcnow():
    return datetime.now(timezone.utc)

class CameraRegistry(Base):
    __tablename__ = "camera_registry"

    camera_id = Column(String, primary_key=True, index=True)
    location_lat = Column(Float, nullable=True)
    location_lon = Column(Float, nullable=True)
    fov_polygon = Column(JSON, nullable=True)
    geo_fence_polygon = Column(JSON, nullable=True)
    calibration_reference_points = Column(JSON, nullable=True)
    trust_score = Column(Float, default=1.0)
    status = Column(String, default="online")
    last_tamper_check = Column(DateTime, default=utcnow)
    stream_url = Column(String, nullable=True)

    entities = relationship("EntityLog", back_populates="camera")


class EntityLog(Base):
    __tablename__ = "entity_log"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, ForeignKey("camera_registry.camera_id"), index=True, nullable=True)
    timestamp = Column(DateTime, default=utcnow, index=True)
    entity_type = Column(String, index=True)  # human | vehicle | animal
    upper_color = Column(String, nullable=True)
    lower_color = Column(String, nullable=True)
    height_cm = Column(Float, nullable=True)
    gender = Column(String, nullable=True)  # male | female | neutral
    plate_text = Column(String, nullable=True)
    vehicle_type = Column(String, nullable=True)  # car | truck | bus | motorcycle
    direction = Column(String, nullable=True)  # North | South | East | West | North-East ...
    speed_kmh = Column(Float, nullable=True)  # estimated speed in km/h
    face_name = Column(String, nullable=True)  # Matched suspect name or Unidentified
    skin_tone = Column(String, nullable=True)  # fair | wheatish | medium | dark
    location_lat = Column(Float, nullable=True)
    location_lon = Column(Float, nullable=True)
    trajectory_id = Column(String, nullable=True)
    is_alert = Column(Boolean, default=False, index=True)
    alert_type = Column(String, nullable=True)  # geo_fence | behavior | tamper | correlated
    confidence_score = Column(Float, default=0.0)
    clip_path = Column(String, nullable=True)
    thumbnail_path = Column(String, nullable=True)
    plate_thumbnail_path = Column(String, nullable=True)
    clip_sha256 = Column(String, nullable=True)
    notarization_token = Column(String, nullable=True)
    retention_tier = Column(String, default="passive")  # passive | protected
    deleted_manually = Column(Boolean, default=False)
    deletion_audit = Column(JSON, nullable=True)
    rule_fired = Column(String, nullable=True)
    acknowledged = Column(Boolean, default=False)
    is_low_light = Column(Boolean, default=False, index=True)
    posture = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_entity_cam_ts", "camera_id", "timestamp"),
        Index("ix_entity_alert_ack", "is_alert", "acknowledged"),
        Index("ix_entity_lowlight_ts", "is_low_light", "timestamp"),
    )

    camera = relationship("CameraRegistry", back_populates="entities")
    false_flags = relationship("FalseFlagLog", back_populates="entity")
    exports = relationship("ExportLog", back_populates="entity")
    tombstones = relationship("MediaTombstoneLog", back_populates="entity")



class FalseFlagLog(Base):
    __tablename__ = "false_flag_log"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    entity_log_id = Column(String, ForeignKey("entity_log.id"), nullable=False)
    marked_by = Column(String, nullable=False)
    marked_at = Column(DateTime, default=utcnow)
    moved_to_hard_negatives = Column(Boolean, default=True)
    notes = Column(String, nullable=True)

    entity = relationship("EntityLog", back_populates="false_flags")


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # commander | operator | admin
    created_at = Column(DateTime, default=utcnow)
    active = Column(Boolean, default=True)

    exports = relationship("ExportLog", back_populates="user")
    audits = relationship("AuditLog", back_populates="user")


class ExportLog(Base):
    __tablename__ = "export_log"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    entity_log_id = Column(String, ForeignKey("entity_log.id"), nullable=False)
    exported_by = Column(String, ForeignKey("users.id"), nullable=False)
    exported_at = Column(DateTime, default=utcnow)
    payload_hash = Column(String, nullable=False)
    channel = Column(String, default="local_bundle")

    entity = relationship("EntityLog", back_populates="exports")
    user = relationship("User", back_populates="exports")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)
    target_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=utcnow)
    details = Column(JSON, nullable=True)

    user = relationship("User", back_populates="audits")


class ArchiveEpochIndex(Base):
    """
    Cryptographic Merkle Root Rollup for legal evidential integrity.
    Rolled up periodically (e.g. hourly) per camera.
    """
    __tablename__ = "archive_epoch_index"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    epoch_id = Column(String, unique=True, index=True, nullable=False)
    camera_id = Column(String, ForeignKey("camera_registry.camera_id"), index=True, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    merkle_root = Column(String, nullable=False)
    frame_count = Column(Integer, default=0)
    alert_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)


class MediaTombstoneLog(Base):
    """
    Immutable tombstone record created when clips are unlinked
    during 92% emergency FIFO purge to recover disk space while preserving
    evidentiary Merkle root and audit history.
    """
    __tablename__ = "media_tombstone_log"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tombstone_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    entity_id = Column(String, ForeignKey("entity_log.id"), nullable=False, index=True)
    camera_id = Column(String, nullable=False, index=True)
    original_clip_path = Column(String, nullable=False)
    clip_sha256 = Column(String, nullable=False)
    evicted_at = Column(DateTime, default=utcnow)
    disk_usage_pct_at_eviction = Column(Float, nullable=False)
    metadata_preserved = Column(JSON, nullable=True)

    entity = relationship("EntityLog", back_populates="tombstones")


class PlateWatchlist(Base):
    """
    ANPR Target License Plate Watchlist / Hotlist.
    Vehicles matching these registered plates trigger automated tactical alerts.
    """
    __tablename__ = "plate_watchlist"

    plate_number = Column(String, primary_key=True, index=True)
    threat_level = Column(String, default="HIGH")  # CRITICAL | HIGH | WATCHLIST
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    active = Column(Boolean, default=True)

