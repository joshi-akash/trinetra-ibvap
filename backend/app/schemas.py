from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# --- Auth Schemas ---
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: str
    username: str

class UserOut(BaseModel):
    id: str
    username: str
    role: str
    active: bool

# --- Camera Schemas ---
class CameraOut(BaseModel):
    camera_id: str
    location: Optional[Dict[str, float]] = None
    fov_polygon: Optional[List[List[float]]] = None
    geo_fence_polygon: Optional[List[List[float]]] = None
    calibration_reference_points: Optional[Dict[str, Any]] = None
    trust_score: float
    status: str
    last_tamper_check: Optional[datetime] = None
    stream_url: Optional[str] = None

class CameraCreate(BaseModel):
    camera_id: str
    location_lat: Optional[float] = 29.9457
    location_lon: Optional[float] = 78.1642
    status: Optional[str] = "online"
    trust_score: Optional[float] = 0.95
    geo_fence_polygon: Optional[List[List[float]]] = None
    stream_url: Optional[str] = None

class StreamUpdateRequest(BaseModel):
    stream_url: Optional[str] = None

class CalibrationPoint(BaseModel):
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float

class CalibrationRequest(BaseModel):
    points: List[CalibrationPoint]

class GeoFencePolygon(BaseModel):
    coordinates: List[List[float]]

# --- Entity Log Schemas ---
class EntityLogOut(BaseModel):
    id: str
    camera_id: Optional[str] = None
    timestamp: datetime
    entity_type: Optional[str] = None
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    height_cm: Optional[float] = None
    gender: Optional[str] = None
    plate_text: Optional[str] = None
    vehicle_type: Optional[str] = None
    direction: Optional[str] = None
    speed_kmh: Optional[float] = None
    face_name: Optional[str] = None
    skin_tone: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    trajectory_id: Optional[str] = None
    is_alert: bool
    alert_type: Optional[str] = None
    confidence_score: float
    clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    retention_tier: str
    deleted_manually: bool
    rule_fired: Optional[str] = None
    acknowledged: bool = False
    is_low_light: bool = False
    posture: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

# --- False Flag & Delete ---
class FalseFlagRequest(BaseModel):
    moved_to_hard_negatives: bool = True
    notes: Optional[str] = None

class FalseFlagOut(BaseModel):
    id: str
    entity_log_id: str
    marked_by: str
    marked_at: datetime
    moved_to_hard_negatives: bool
    notes: Optional[str] = None

# --- Alert Schemas ---
class AlertDetailOut(BaseModel):
    id: str
    camera_id: str
    timestamp: datetime
    alert_type: str
    rule_fired: Optional[str] = None
    confidence_score: float
    thumbnail_url: Optional[str] = None
    clip_url: Optional[str] = None
    acknowledged: bool
    entity_details: Optional[EntityLogOut] = None

class AcknowledgeRequest(BaseModel):
    notes: Optional[str] = None

# --- Export Schemas ---
class ExportRequest(BaseModel):
    include_clip: bool = True
    channel: str = "local_bundle"

class ExportResponse(BaseModel):
    export_id: str
    entity_log_id: str
    payload_hash: str
    download_url: str
    exported_at: datetime
    channel: str

# --- Forensic Search Schemas ---
class SearchRequest(BaseModel):
    query: Optional[str] = None
    camera_id: Optional[str] = None
    entity_type: Optional[str] = None
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    posture: Optional[str] = None
    direction: Optional[str] = None
    is_low_light: Optional[bool] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    alert_only: Optional[bool] = None
    limit: int = 50
    offset: int = 0


class SearchResponse(BaseModel):
    parsed_filters: Dict[str, Any]
    total_count: int
    results: List[EntityLogOut]

# --- Contract 1: FrameAnalysis Ingestion (From Person A & B) ---
class TamperSignal(BaseModel):
    is_tampered: bool = False
    laplacian_variance: float = 500.0
    histogram_flag: bool = False
    reference_point_drift: bool = False

class FrameQuality(BaseModel):
    low_light: bool = False
    enhanced: bool = False
    tamper_signal: TamperSignal = Field(default_factory=TamperSignal)

class FaceMatch(BaseModel):
    suspect_id: str
    confidence: float

class EntityAttributes(BaseModel):
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    height_cm: Optional[float] = None
    gender: Optional[str] = None
    plate_text: Optional[str] = None
    vehicle_type: Optional[str] = None
    direction: Optional[str] = None
    speed_kmh: Optional[float] = None
    face_name: Optional[str] = None
    skin_tone: Optional[str] = None
    face_match: Optional[FaceMatch] = None
    posture: Optional[str] = None
    props: List[str] = Field(default_factory=list)

class LocationPoint(BaseModel):
    lat: float
    lon: float

class EntityDetection(BaseModel):
    track_id: str
    entity_type: str  # human | vehicle | animal
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    confidence: float
    attributes: EntityAttributes = Field(default_factory=EntityAttributes)
    location: LocationPoint

class FrameAnalysis(BaseModel):
    camera_id: str
    timestamp: datetime
    frame_ref: str
    frame_quality: FrameQuality = Field(default_factory=FrameQuality)
    entities: List[EntityDetection] = Field(default_factory=list)

# --- Generic Utility Schemas ---
class GenericStatusResponse(BaseModel):
    status: str
    message: str

class FrameDetectRequest(BaseModel):
    image: str
    timestamp: Optional[str] = None
    width: Optional[int] = 640
    height: Optional[int] = 360

class DetectedEntityOut(BaseModel):
    entity_type: str
    bbox: List[int]
    confidence: float
    track_id: Optional[Union[int, str]] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

class FrameDetectResponse(BaseModel):
    camera_id: str
    entities: List[DetectedEntityOut] = Field(default_factory=list)
    frame_width: int = 640
    frame_height: int = 360

# --- Continuous Model Training Schemas ---
class TrainingStartRequest(BaseModel):
    stream_url: Optional[str] = None
    epochs: int = 3
    max_frames: int = 40
    base_weights: str = "models/yolov8n.pt"
    enhance_low_light: bool = True

class TrainingStatusResponse(BaseModel):
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    total_frames: int
    annotated_boxes: int
    box_loss: float
    cls_loss: float
    weights_path: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    duration_seconds: float = 0.0

class ApplyWeightsRequest(BaseModel):
    weights_path: Optional[str] = None


