from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import CameraRegistry, User
from backend.app.schemas import CameraOut, CameraCreate, CalibrationRequest, GenericStatusResponse
from backend.app.auth.jwt_auth import get_current_user

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
            last_tamper_check=c.last_tamper_check
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
        db.commit()
        return GenericStatusResponse(status="success", message=f"Camera {req.camera_id} updated successfully")

    camera = CameraRegistry(
        camera_id=req.camera_id,
        location_lat=req.location_lat,
        location_lon=req.location_lon,
        status=req.status or "online",
        trust_score=req.trust_score or 0.95,
        geo_fence_polygon=req.geo_fence_polygon
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
    db.delete(camera)
    db.commit()
    return GenericStatusResponse(status="success", message=f"Camera {camera_id} removed from registry")

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

