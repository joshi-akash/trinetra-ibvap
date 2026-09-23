import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.database import get_db
from backend.app.models import EntityLog, PlateWatchlist, User
from backend.app.auth.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/anpr", tags=["ANPR & Vehicle Intelligence"])


class WatchlistPlateCreate(BaseModel):
    plate_number: str = Field(..., min_length=3, max_length=24, description="License plate string (e.g. DL01AB1234)")
    threat_level: str = Field("HIGH", description="CRITICAL | HIGH | WATCHLIST")
    notes: Optional[str] = Field(None, description="Suspect details or briefing notes")


class WatchlistPlateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    plate_number: str
    threat_level: str
    notes: Optional[str] = None
    created_at: datetime
    active: bool


class PlateSighting(BaseModel):
    id: str
    timestamp: datetime
    camera_id: Optional[str] = None
    plate_text: Optional[str] = None
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    speed_kmh: Optional[float] = None
    direction: Optional[str] = None
    is_alert: bool = False
    alert_type: Optional[str] = None
    thumbnail_path: Optional[str] = None
    plate_thumbnail_path: Optional[str] = None
    clip_path: Optional[str] = None


class TrackPlateAssignRequest(BaseModel):
    camera_id: str
    track_id: str
    plate_text: str


@router.get("/watchlist", response_model=List[WatchlistPlateResponse])
def get_plate_watchlist(db: Session = Depends(get_db)):
    """Retrieve all active flagged license plates in the ANPR hotlist."""
    return db.query(PlateWatchlist).order_by(desc(PlateWatchlist.created_at)).all()


@router.post("/watchlist", response_model=WatchlistPlateResponse)
def add_plate_to_watchlist(
    req: WatchlistPlateCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Register a new flagged vehicle license plate for automated tactical breach alerting."""
    clean_plate = req.plate_number.replace(" ", "").replace("-", "").upper()
    existing = db.query(PlateWatchlist).filter(PlateWatchlist.plate_number == clean_plate).first()
    if existing:
        existing.threat_level = req.threat_level
        existing.notes = req.notes
        existing.active = True
        db.commit()
        db.refresh(existing)
        return existing

    new_entry = PlateWatchlist(
        plate_number=clean_plate,
        threat_level=req.threat_level,
        notes=req.notes,
        created_at=datetime.now(timezone.utc),
        active=True
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry


@router.delete("/watchlist/{plate_number}")
def remove_plate_from_watchlist(
    plate_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Remove a vehicle license plate from the ANPR tactical watchlist."""
    clean_plate = plate_number.replace(" ", "").replace("-", "").upper()
    item = db.query(PlateWatchlist).filter(PlateWatchlist.plate_number == clean_plate).first()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plate not found in watchlist")
    db.delete(item)
    db.commit()
    return {"status": "success", "message": f"Plate {clean_plate} removed from watchlist"}


@router.get("/search", response_model=List[PlateSighting])
def search_plate_records(
    query: str = Query(..., min_length=2, description="Plate number or partial substring (e.g. DL01, HR26)"),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    """Search historical CCTV sightings for a specific license plate across all cameras."""
    clean_query = query.replace(" ", "").replace("-", "")
    records = (
        db.query(EntityLog)
        .filter(EntityLog.entity_type == "vehicle")
        .filter(EntityLog.plate_text.ilike(f"%{clean_query}%"))
        .order_by(desc(EntityLog.timestamp))
        .limit(limit)
        .all()
    )

    out = []
    for r in records:
        out.append(PlateSighting(
            id=r.id,
            timestamp=r.timestamp,
            camera_id=r.camera_id,
            plate_text=r.plate_text,
            vehicle_type=r.vehicle_type,
            vehicle_color=r.upper_color,
            speed_kmh=r.speed_kmh,
            direction=r.direction,
            is_alert=r.is_alert,
            alert_type=r.alert_type,
            thumbnail_path=r.thumbnail_path,
            plate_thumbnail_path=getattr(r, "plate_thumbnail_path", None),
            clip_path=r.clip_path
        ))
    return out


@router.post("/assign")
def assign_track_plate(req: TrackPlateAssignRequest):
    """Manually assign or override a plate string for an active vehicle track in memory."""
    from ai_detection import _TRACK_PLATE_MEMORY
    clean_plate = req.plate_text.replace(" ", "").replace("-", "").upper()
    cache_key = (str(req.camera_id), str(req.track_id))
    _TRACK_PLATE_MEMORY[cache_key] = clean_plate
    return {"status": "success", "camera_id": req.camera_id, "track_id": req.track_id, "plate_text": clean_plate}
