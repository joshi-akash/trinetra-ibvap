import pytest
import uuid
import zipfile
import json
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.database import Base, get_db
from backend.app.auth.jwt_auth import get_password_hash
from backend.app.models import User, CameraRegistry, EntityLog, AuditLog, FalseFlagLog, ExportLog

# Setup an isolated test in-memory SQLite database
TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    try:
        # Seed test users
        users = [
            User(
                id=str(uuid.UUID("a0000000-0000-0000-0000-000000000001")),
                username="admin_user",
                password_hash=get_password_hash("admin123"),
                role="admin",
                active=True
            ),
            User(
                id=str(uuid.UUID("c0000000-0000-0000-0000-000000000001")),
                username="commander_user",
                password_hash=get_password_hash("commander123"),
                role="commander",
                active=True
            ),
            User(
                id=str(uuid.UUID("e0000000-0000-0000-0000-000000000001")),
                username="operator_user",
                password_hash=get_password_hash("operator123"),
                role="operator",
                active=True
            )
        ]
        db.add_all(users)

        # Seed test camera with initial geofence polygon (box around lat 29.945-29.947, lon 78.164-78.166)
        cam = CameraRegistry(
            camera_id="CAM-01",
            location_lat=29.9460,
            location_lon=78.1650,
            trust_score=0.98,
            status="online",
            geo_fence_polygon=[
                [78.1640, 29.9450],
                [78.1660, 29.9450],
                [78.1660, 29.9470],
                [78.1640, 29.9470],
                [78.1640, 29.9450]
            ]
        )
        db.add(cam)
        db.commit()
    finally:
        db.close()

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

@pytest.fixture(scope="module")
def commander_token(client):
    res = client.post("/api/auth/login", json={"username": "commander_user", "password": "commander123"})
    assert res.status_code == 200
    return res.json()["access_token"]

@pytest.fixture(scope="module")
def operator_token(client):
    res = client.post("/api/auth/login", json={"username": "operator_user", "password": "operator123"})
    assert res.status_code == 200
    return res.json()["access_token"]

@pytest.fixture(scope="module")
def admin_token(client):
    res = client.post("/api/auth/login", json={"username": "admin_user", "password": "admin123"})
    assert res.status_code == 200
    return res.json()["access_token"]


# --- 1. Health Endpoint ---
def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["sovereign_offline_ready"] is True


# --- 2. Auth Endpoints & Password Verification ---
def test_login_success(client):
    res = client.post("/api/auth/login", json={"username": "commander_user", "password": "commander123"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["role"] == "commander"
    assert data["token_type"] == "bearer"

def test_login_invalid_password(client):
    res = client.post("/api/auth/login", json={"username": "commander_user", "password": "wrongpassword"})
    assert res.status_code == 401


# --- 3. Cameras & Ground-Plane Calibration ---
def test_list_cameras(client, commander_token):
    res = client.get("/api/cameras", headers={"Authorization": f"Bearer {commander_token}"})
    assert res.status_code == 200
    cameras = res.json()
    assert len(cameras) >= 1
    assert any(c["camera_id"] == "CAM-01" for c in cameras)

def test_calibrate_camera(client, commander_token):
    points = [
        {"pixel_x": 100.0, "pixel_y": 200.0, "world_x": 0.0, "world_y": 0.0},
        {"pixel_x": 300.0, "pixel_y": 200.0, "world_x": 10.0, "world_y": 0.0},
        {"pixel_x": 300.0, "pixel_y": 400.0, "world_x": 10.0, "world_y": 10.0},
        {"pixel_x": 100.0, "pixel_y": 400.0, "world_x": 0.0, "world_y": 10.0}
    ]
    res = client.post(
        "/api/cameras/CAM-01/calibrate",
        headers={"Authorization": f"Bearer {commander_token}"},
        json={"points": points}
    )
    assert res.status_code == 200
    assert res.json()["status"] == "success"

def test_calibrate_nonexistent_camera(client, commander_token):
    res = client.post(
        "/api/cameras/NONEXISTENT/calibrate",
        headers={"Authorization": f"Bearer {commander_token}"},
        json={"points": [{"pixel_x": 0, "pixel_y": 0, "world_x": 0, "world_y": 0}]}
    )
    assert res.status_code == 404


# --- 4. Geo-Fence Management ---
def test_get_geofence(client, commander_token):
    res = client.get("/api/geofence/CAM-01", headers={"Authorization": f"Bearer {commander_token}"})
    assert res.status_code == 200
    coords = res.json()["coordinates"]
    assert len(coords) >= 4

def test_update_geofence_valid(client, commander_token):
    new_coords = [
        [78.1630, 29.9440],
        [78.1670, 29.9440],
        [78.1670, 29.9480],
        [78.1630, 29.9480],
        [78.1630, 29.9440]
    ]
    res = client.put(
        "/api/geofence/CAM-01",
        headers={"Authorization": f"Bearer {commander_token}"},
        json={"coordinates": new_coords}
    )
    assert res.status_code == 200
    assert "updated" in res.json()["message"]

def test_update_geofence_invalid_vertices(client, commander_token):
    res = client.put(
        "/api/geofence/CAM-01",
        headers={"Authorization": f"Bearer {commander_token}"},
        json={"coordinates": [[78.1630, 29.9440], [78.1670, 29.9440]]}
    )
    assert res.status_code == 400


# --- 5. AI Ingestion & Bifurcation Path ---
def test_ingest_passive_frame(client):
    """Movement outside geofence -> Passive Log (FR-ALR-01)"""
    payload = {
        "camera_id": "CAM-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_ref": "frame_passive_01.jpg",
        "frame_quality": {
            "low_light": False,
            "enhanced": False,
            "tamper_signal": {
                "is_tampered": False,
                "laplacian_variance": 550.0
            }
        },
        "entities": [
            {
                "track_id": "trk-passive-1",
                "entity_type": "human",
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "confidence": 0.86,
                "attributes": {
                    "upper_color": "green",
                    "lower_color": "khaki",
                    "posture": "standing",
                    "props": []
                },
                "location": {"lat": 29.9800, "lon": 78.1900}  # Far outside fence
            }
        ]
    }
    res = client.post("/api/entities/ingest", json=payload)
    assert res.status_code == 200
    assert "0 active alerts, 1 passive logs" in res.json()["message"]

def test_ingest_geofence_breach(client):
    """Crossing into geofence -> Active Alert + WebP + 30s Clip (FR-ALR-02.1)"""
    payload = {
        "camera_id": "CAM-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_ref": "frame_breach_02.jpg",
        "frame_quality": {
            "low_light": False,
            "enhanced": False,
            "tamper_signal": {
                "is_tampered": False,
                "laplacian_variance": 600.0
            }
        },
        "entities": [
            {
                "track_id": "trk-breach-2",
                "entity_type": "human",
                "bbox": [120.0, 80.0, 250.0, 300.0],
                "confidence": 0.93,
                "attributes": {
                    "upper_color": "red",
                    "lower_color": "blue",
                    "posture": "standing",
                    "props": []
                },
                "location": {"lat": 29.9460, "lon": 78.1650}  # Clearly inside fence
            }
        ]
    }
    res = client.post("/api/entities/ingest", json=payload)
    assert res.status_code == 200
    assert "1 active alerts" in res.json()["message"]

def test_ingest_multi_modal_corroboration(client):
    """Armed crouched intruder inside boundary -> Correlated High-Severity Alert (FR-ALR-04)"""
    payload = {
        "camera_id": "CAM-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_ref": "frame_corroborated_03.jpg",
        "frame_quality": {
            "low_light": True,
            "enhanced": True,
            "tamper_signal": {
                "is_tampered": False,
                "laplacian_variance": 480.0
            }
        },
        "entities": [
            {
                "track_id": "trk-threat-3",
                "entity_type": "human",
                "bbox": [150.0, 100.0, 300.0, 400.0],
                "confidence": 0.95,
                "attributes": {
                    "upper_color": "black",
                    "lower_color": "camo",
                    "posture": "crouching",
                    "props": ["weapon"]
                },
                "location": {"lat": 29.9460, "lon": 78.1650}  # Inside fence
            }
        ]
    }
    res = client.post("/api/entities/ingest", json=payload)
    assert res.status_code == 200
    assert "1 active alerts" in res.json()["message"]

def test_ingest_camera_tamper(client):
    """Camera blinding/spray -> Tamper Alert (FR-TMP-01)"""
    payload = {
        "camera_id": "CAM-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_ref": "frame_tamper_04.jpg",
        "frame_quality": {
            "low_light": False,
            "enhanced": False,
            "tamper_signal": {
                "is_tampered": True,
                "laplacian_variance": 35.0,
                "histogram_flag": True
            }
        },
        "entities": []
    }
    res = client.post("/api/entities/ingest", json=payload)
    assert res.status_code == 200
    assert "1 active alerts" in res.json()["message"]


# --- 6. Alerts Feed & Acknowledgement ---
def test_get_active_alerts_and_acknowledge(client, commander_token):
    # Fetch active alerts
    res = client.get("/api/alerts/active", headers={"Authorization": f"Bearer {commander_token}"})
    assert res.status_code == 200
    alerts = res.json()
    assert len(alerts) >= 2  # At least breach + corroborated alerts

    alert_id = alerts[0]["id"]
    # Acknowledge the first alert
    ack_res = client.post(
        f"/api/alerts/{alert_id}/acknowledge",
        headers={"Authorization": f"Bearer {commander_token}"},
        json={"notes": "Investigated by sector patrol"}
    )
    assert ack_res.status_code == 200
    assert "acknowledged" in ack_res.json()["message"]

    # Verify acknowledged alert is no longer in active alerts feed
    refetched = client.get("/api/alerts/active", headers={"Authorization": f"Bearer {commander_token}"}).json()
    assert all(a["id"] != alert_id for a in refetched)


# --- 7. False Flag Active Feedback Loop ---
def test_false_flag_marking(client, commander_token):
    # Fetch an existing entity
    entities = client.get("/api/entities", headers={"Authorization": f"Bearer {commander_token}"}).json()
    assert len(entities) > 0
    target_entity = entities[0]

    # Mark as false flag
    res = client.post(
        f"/api/entities/{target_entity['id']}/false-flag",
        headers={"Authorization": f"Bearer {commander_token}"},
        json={"moved_to_hard_negatives": True, "notes": "Tree shadow mistaken for crouching human"}
    )
    assert res.status_code == 200
    flag_data = res.json()
    assert flag_data["entity_log_id"] == target_entity["id"]
    assert flag_data["moved_to_hard_negatives"] is True


# --- 8. Offline HQ Export Package with RBAC & SHA-256 ---
def test_export_rbac_forbidden_for_operator(client, operator_token, commander_token):
    entities = client.get("/api/entities", headers={"Authorization": f"Bearer {commander_token}"}).json()
    entity_id = entities[0]["id"]

    # Operator role cannot export (FR-EXP-03)
    res = client.post(
        f"/api/entities/{entity_id}/export",
        headers={"Authorization": f"Bearer {operator_token}"},
        json={"include_clip": True, "channel": "local_bundle"}
    )
    assert res.status_code == 403

def test_export_allowed_for_commander(client, commander_token):
    entities = client.get("/api/entities", headers={"Authorization": f"Bearer {commander_token}"}).json()
    entity_id = entities[0]["id"]

    res = client.post(
        f"/api/entities/{entity_id}/export",
        headers={"Authorization": f"Bearer {commander_token}"},
        json={"include_clip": True, "channel": "local_bundle"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "payload_hash" in data
    assert len(data["payload_hash"]) == 64  # Valid SHA-256 length
    assert "download_url" in data


# --- 9. Manual Audit Deletion (FR-RET-02) ---
def test_manual_deletion_rbac_and_audit(client, operator_token, commander_token):
    entities = client.get("/api/entities", headers={"Authorization": f"Bearer {commander_token}"}).json()
    entity_id = entities[-1]["id"]

    # Operator cannot delete
    del_res = client.delete(
        f"/api/entities/{entity_id}?reason=False+alarm",
        headers={"Authorization": f"Bearer {operator_token}"}
    )
    assert del_res.status_code == 403

    # Commander requires reason
    no_reason = client.delete(
        f"/api/entities/{entity_id}",
        headers={"Authorization": f"Bearer {commander_token}"}
    )
    assert no_reason.status_code == 422

    # Commander with reason succeeds
    del_success = client.delete(
        f"/api/entities/{entity_id}?reason=Friendly+sector+inspection+cleared",
        headers={"Authorization": f"Bearer {commander_token}"}
    )
    assert del_success.status_code == 200
    assert "marked deleted" in del_success.json()["message"]

    # Verify deleted record is omitted from standard query
    active_entities = client.get("/api/entities", headers={"Authorization": f"Bearer {commander_token}"}).json()
    assert all(e["id"] != entity_id for e in active_entities)


# --- 10. Vernacular Forensic Search ---
def test_forensic_search_vernacular(client, commander_token):
    # Search for red shirt person using Hindi query "laal shirt aadmi"
    res = client.post(
        "/api/search",
        headers={"Authorization": f"Bearer {commander_token}"},
        json={"query": "laal shirt aadmi"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["parsed_filters"]["resolved"]["color"] == "red"
    assert data["parsed_filters"]["resolved"]["entity_type"] == "human"
    # Should find our red shirt intruder
    assert data["total_count"] >= 1
    assert any(r["upper_color"] == "red" for r in data["results"])
