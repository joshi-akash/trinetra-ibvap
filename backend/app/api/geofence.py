from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import CameraRegistry, User
from backend.app.schemas import GeoFencePolygon, GenericStatusResponse
from backend.app.auth.jwt_auth import get_current_user

router = APIRouter(prefix="/api/geofence", tags=["GeoFence"])

@router.get("/{camera_id}", response_model=GeoFencePolygon)
def get_geofence(
    camera_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    coords = camera.geo_fence_polygon or []
    return GeoFencePolygon(coordinates=coords)

@router.put("/{camera_id}", response_model=GenericStatusResponse)
def update_geofence(
    camera_id: str,
    req: GeoFencePolygon,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if len(req.coordinates) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A geo-fence polygon must contain at least 3 vertices."
        )

    camera = db.query(CameraRegistry).filter(CameraRegistry.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    camera.geo_fence_polygon = req.coordinates
    db.commit()

    return GenericStatusResponse(
        status="success",
        message=f"Geo-fence polygon updated for {camera_id} ({len(req.coordinates)} vertices)"
    )
