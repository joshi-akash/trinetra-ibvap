import io
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.database import Base, get_db
from backend.app.models import User, CameraRegistry
from backend.app.auth.jwt_auth import get_password_hash

TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSession()
    try:
        user = User(
            id="test-commander-id",
            username="test_commander",
            password_hash=get_password_hash("pass123"),
            role="commander",
            active=True
        )
        db.add(user)
        cam = CameraRegistry(
            camera_id="CAM-01",
            location_lat=29.9457,
            location_lon=78.1642,
            trust_score=0.98,
            status="online",
            stream_url="/footage/cctv_sample.mp4"
        )
        db.add(cam)
        db.commit()
    finally:
        db.close()

    def override_get_db():
        session = TestingSession()
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
def auth_headers(client):
    res = client.post("/api/auth/login", json={"username": "test_commander", "password": "pass123"})
    assert res.status_code == 200
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_static_footage_presets_accessible(client):
    """Verify all 3 local tactical presets are served via /footage/."""
    for preset in ["cctv_sample.mp4", "highway_patrol.mp4", "night_patrol.mp4"]:
        res = client.get(f"/footage/{preset}")
        assert res.status_code == 200, f"Preset {preset} must be accessible via HTTP 200"
        assert len(res.content) > 1000, f"Preset {preset} content must not be empty"

def test_upload_footage_auto_registers_new_camera(client, auth_headers):
    """Upload to an unassigned camera ID (e.g. CAM-99) must auto-register the camera."""
    fake_video = io.BytesIO(b"fake mp4 video content test data header")
    response = client.post(
        "/api/cameras/CAM-99/upload-footage",
        files={"file": ("border patrol night test.mp4", fake_video, "video/mp4")},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "border_patrol_night_test" in data["message"]

    # Verify camera exists and has stream_url set
    cam_res = client.get("/api/cameras", headers=auth_headers)
    assert cam_res.status_code == 200
    cams = cam_res.json()
    cam99 = next((c for c in cams if c["camera_id"] == "CAM-99"), None)
    assert cam99 is not None
    assert "/footage/" in cam99["stream_url"]

def test_low_light_frame_ingestion_and_bifurcation(client):
    """Ingest a zero-light night infiltration frame (11.4 lux) and verify alert creation."""
    payload = {
        "camera_id": "CAM-01",
        "timestamp": "2026-09-13T20:00:00Z",
        "frame_ref": "frame_night_01.jpg",
        "frame_quality": {
            "low_light": True,
            "enhanced": True,
            "tau_visibility": 0.42,
            "tamper_signal": {
                "is_tampered": False,
                "laplacian_variance": 42.0,
                "histogram_flag": False,
                "reference_point_drift": False,
                "static_freeze": False
            }
        },
        "entities": [
            {
                "track_id": "night-suspect-01",
                "entity_type": "human",
                "bbox": [100, 150, 250, 400],
                "confidence": 0.94,
                "attributes": {
                    "upper_color": "black",
                    "lower_color": "dark_olive",
                    "posture": "crouching",
                    "skin_tone": "wheatish",
                    "face_name": "Akash (SUSP-001)",
                    "props": ["weapon"]
                },
                "location": {"lat": 29.9458, "lon": 78.1643}
            }
        ]
    }
    res = client.post("/api/entities/ingest", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "Processed frame" in data["message"]

def test_stream_update_presets_and_clearing(client, auth_headers):
    """Verify updating camera stream with presets, empty strings, and auto-registration."""
    # 1. Update CAM-01 to highway patrol preset
    res1 = client.put("/api/cameras/CAM-01/stream", json={"stream_url": "/footage/highway_patrol.mp4"}, headers=auth_headers)
    assert res1.status_code == 200
    assert res1.json()["status"] == "success"

    # 2. Revert CAM-01 to radar mode via empty string
    res2 = client.put("/api/cameras/CAM-01/stream", json={"stream_url": ""}, headers=auth_headers)
    assert res2.status_code == 200
    assert res2.json()["status"] == "success"

    # 3. Auto-create unregistered camera when updating stream
    res3 = client.put("/api/cameras/CAM-NEW-42/stream", json={"stream_url": "/footage/night_patrol.mp4"}, headers=auth_headers)
    assert res3.status_code == 200
    assert res3.json()["status"] == "success"

    # Verify CAM-NEW-42 exists
    cam_res = client.get("/api/cameras", headers=auth_headers)
    assert any(c["camera_id"] == "CAM-NEW-42" and c["stream_url"] == "/footage/night_patrol.mp4" for c in cam_res.json())

def test_skylinewebcams_and_hls_stream_resolution(client, auth_headers):
    """Verify resolve_live_stream_url and update_camera_stream for online webcam links."""
    from backend.app.api.cameras import resolve_live_stream_url

    # 1. Direct streams should remain untouched
    direct = "https://cph-p2p-msl.akamaized.net/hls/live/2000341/test/master.m3u8"
    assert resolve_live_stream_url(direct) == direct

    # 2. None / empty strings
    assert resolve_live_stream_url(None) is None
    assert resolve_live_stream_url("") is None

    # 3. Test mock resolution of SkylineWebcams URL
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<script>var player=new Clappr.Player({source:'livee.m3u8?a=mocktoken12345'});</script>"
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        page_url = "https://www.skylinewebcams.com/en/webcam/thailand/surat-thani/ko-samui/lamai.html"
        resolved = resolve_live_stream_url(page_url)
        assert resolved == "https://hd-auth.skylinewebcams.com/live.m3u8?a=mocktoken12345"

        # Update camera stream with SkylineWebcams URL
        res = client.put("/api/cameras/CAM-01/stream", json={"stream_url": page_url}, headers=auth_headers)
        assert res.status_code == 200
        assert "Live HLS stream auto-resolved" in res.json()["message"]

        # Verify CAM-01 stored resolved URL
        cam_res = client.get("/api/cameras", headers=auth_headers)
        c01 = next(c for c in cam_res.json() if c["camera_id"] == "CAM-01")
        assert c01["stream_url"] == "https://hd-auth.skylinewebcams.com/live.m3u8?a=mocktoken12345"

def test_detect_camera_frame_endpoint(client):
    """Verify POST /api/cameras/{camera_id}/detect-frame decodes base64 and returns detections."""
    import base64
    import numpy as np
    import cv2

    img = np.zeros((360, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buf).decode("utf-8")

    res = client.post("/api/cameras/CAM-01/detect-frame", json={"image": b64_str, "width": 640, "height": 360})
    assert res.status_code == 200
    data = res.json()
    assert data["camera_id"] == "CAM-01"
    assert "entities" in data
    assert data["frame_width"] == 640
    assert data["frame_height"] == 360


