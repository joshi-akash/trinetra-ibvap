import pytest
import uuid
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.database import Base, get_db
from backend.app.auth.jwt_auth import get_password_hash
from backend.app.models import User, CameraRegistry, EntityLog
from backend.app.schemas import (
    FrameAnalysis, FrameQuality, TamperSignal,
    EntityDetection, EntityAttributes, LocationPoint
)
from backend.app.rule_engine.bifurcation import bifurcation_engine
from backend.app.rule_engine.condition_wiring import condition_wiring
from backend.app.search_service.synonym_dictionary import synonym_dictionary

TEST_DB_URL = "sqlite:///:memory:"
test_engine_night = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionNight = sessionmaker(autocommit=False, autoflush=False, bind=test_engine_night)

@pytest.fixture(scope="module", autouse=True)
def setup_night_test_db():
    Base.metadata.create_all(bind=test_engine_night)
    db = TestingSessionNight()
    try:
        user = User(
            id=str(uuid.UUID("b0000000-0000-0000-0000-000000000001")),
            username="commander_night",
            password_hash=get_password_hash("commander123"),
            role="commander",
            active=True
        )
        db.add(user)

        cam = CameraRegistry(
            camera_id="CAM-NIGHT-01",
            location_lat=29.9460,
            location_lon=78.1650,
            trust_score=1.0,
            status="online",
            geo_fence_polygon=[[78.1640, 29.9450], [78.1660, 29.9450], [78.1660, 29.9470], [78.1640, 29.9470]]
        )
        db.add(cam)
        db.commit()
    finally:
        db.close()

    def override_get_db():
        session = TestingSessionNight()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine_night)

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_night_tamper_avoidance():
    """
    In pitch darkness, natural Laplacian variance drops.
    A variance of 45.0 in normal daylight triggers tamper,
    but in low_light mode it must NOT trigger a false tamper alarm.
    """
    # 1. Daylight frame with variance=45.0 -> should trigger tamper alert
    day_tamper = {"is_tampered": False, "laplacian_variance": 45.0}
    is_tampered_day, _ = condition_wiring.is_camera_tampered(day_tamper, is_low_light=False)
    assert is_tampered_day is True, "Daylight variance below 100.0 must trigger tamper alert"

    # 2. Night / low-light frame with variance=45.0 -> must NOT trigger tamper alert (floor is 30.0)
    night_tamper = {"is_tampered": False, "laplacian_variance": 45.0}
    is_tampered_night, _ = condition_wiring.is_camera_tampered(night_tamper, is_low_light=True)
    assert is_tampered_night is False, "Night variance of 45.0 must NOT trigger false tamper alert"

    # 3. Night frame with complete occlusion (variance=15.0 < 30.0) -> must still trigger tamper
    night_occlusion = {"is_tampered": False, "laplacian_variance": 15.0}
    is_tampered_occ, _ = condition_wiring.is_camera_tampered(night_occlusion, is_low_light=True)
    assert is_tampered_occ is True, "Severe night blur below 30.0 must still trigger tamper"

def test_night_stealth_intrusion_escalation():
    """
    In low-light conditions, an entity crouching inside the geofence
    must escalate to 'night_stealth_intrusion' with 'correlated' severity tier.
    """
    geofence = [[78.1640, 29.9450], [78.1660, 29.9450], [78.1660, 29.9470], [78.1640, 29.9470]]
    
    night_frame = FrameAnalysis(
        camera_id="CAM-NIGHT-01",
        timestamp=datetime.now(timezone.utc),
        frame_ref="night_stealth_01.jpg",
        frame_quality=FrameQuality(
            low_light=True,
            enhanced=True,
            tamper_signal=TamperSignal(is_tampered=False, laplacian_variance=45.0)
        ),
        entities=[
            EntityDetection(
                track_id="night-crawl-1",
                entity_type="human",
                bbox=[150, 150, 300, 300],
                confidence=0.88,
                attributes=EntityAttributes(
                    upper_color="black",
                    lower_color="dark",
                    posture="crouching"
                ),
                location=LocationPoint(lat=29.9460, lon=78.1650)  # Inside geofence
            )
        ]
    )

    decisions, camera_alert = bifurcation_engine.evaluate_frame(night_frame, geofence)
    assert camera_alert is None  # No false tamper in dark
    assert len(decisions) == 1
    d = decisions[0]
    assert d.is_alert is True
    assert d.alert_type == "correlated"
    assert "night_stealth_intrusion" in d.rule_fired
    assert d.retention_tier == "protected"

def test_night_vernacular_search():
    """
    Forensic search queries mentioning darkness ('andhere mein aadmi', 'kaali gaadi')
    must automatically resolve is_low_light=True, colors, and license plate.
    """
    parsed = synonym_dictionary.parse_query("andhere mein laal shirt aadmi")
    assert parsed["resolved"].get("is_low_light") is True
    assert parsed["resolved"].get("color") == "red"
    assert parsed["resolved"].get("entity_type") == "human"

    parsed_plate = synonym_dictionary.parse_query("kaali gaadi DL01AB1234")
    assert parsed_plate["resolved"].get("color") == "black"
    assert parsed_plate["resolved"].get("entity_type") == "vehicle"
    assert parsed_plate["resolved"].get("plate_text") == "DL01AB1234"

def test_low_light_and_plate_end_to_end_ingestion(client):
    """
    Full HTTP ingestion of a night frame with vehicle and number plate.
    Verifies is_low_light, posture, and plate_text are preserved in DB.
    """
    login_res = client.post(
        "/api/auth/login",
        json={"username": "commander_night", "password": "commander123"}
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    auth_headers = {"Authorization": f"Bearer {token}"}

    # Ingest a low-light frame with a vehicle and plate number
    ingest_payload = {
        "camera_id": "CAM-NIGHT-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_ref": "test_night_vehicle.jpg",
        "frame_quality": {
            "low_light": True,
            "enhanced": True,
            "tamper_signal": {"is_tampered": False, "laplacian_variance": 55.0}
        },
        "entities": [
            {
                "track_id": "track-veh-night",
                "entity_type": "vehicle",
                "bbox": [50.0, 50.0, 400.0, 300.0],
                "confidence": 0.94,
                "attributes": {
                    "upper_color": "white",
                    "lower_color": "white",
                    "plate_text": "HR26DQ5555"
                },
                "location": {"lat": 29.9800, "lon": 78.1900}  # Outside geofence (passive)
            }
        ]
    }

    ingest_res = client.post("/api/entities/ingest", json=ingest_payload)
    assert ingest_res.status_code == 200

    # Query entities to verify is_low_light and plate_text are persisted
    entities_res = client.get("/api/entities?entity_type=vehicle", headers=auth_headers)
    assert entities_res.status_code == 200
    records = entities_res.json()
    assert len(records) >= 1
    found = next((r for r in records if r.get("plate_text") == "HR26DQ5555"), None)
    assert found is not None
    assert found["is_low_light"] is True
    assert found["entity_type"] == "vehicle"
