import io
import os
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.database import Base, get_db
from backend.app.models import User
from backend.app.auth.jwt_auth import get_password_hash
from ai_detection.training.auto_trainer import AutoTrainer, get_auto_trainer, resolve_stream_source

TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSession()
    try:
        user = User(
            id="test-operator-id",
            username="test_operator",
            password_hash=get_password_hash("pass123"),
            role="operator",
            active=True
        )
        db.add(user)
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


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    login_resp = client.post("/api/auth/login", json={"username": "test_operator", "password": "pass123"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_training_status_endpoint(client, auth_headers):
    """Test telemetry retrieval from /api/training/status."""
    resp = client.get("/api/training/status", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "progress" in data
    assert "logs" in data
    assert isinstance(data["logs"], list)


def test_start_training_missing_parameters(client, auth_headers):
    """Ensure training fails gracefully when no source is specified."""
    resp = client.post(
        "/api/training/start-video-training",
        json={"stream_url": ""},
        headers=auth_headers
    )
    assert resp.status_code == 400


def test_start_training_file_upload(client, auth_headers):
    """Test launching training via multipart video file upload."""
    trainer = get_auto_trainer()
    with patch.object(trainer, "start_training", return_value=True) as mock_start:
        file_bytes = io.BytesIO(b"fake mp4 video binary data for test")
        resp = client.post(
            "/api/training/start-video-training",
            files={"file": ("test_cctv.mp4", file_bytes, "video/mp4")},
            data={"epochs": 2, "max_frames": 20, "base_weights": "models/yolov8n.pt", "enhance_low_light": "true"},
            headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert mock_start.called


def test_start_training_stream_url_json(client, auth_headers):
    """Test launching training via JSON stream URL payload."""
    trainer = get_auto_trainer()
    with patch.object(trainer, "start_training", return_value=True) as mock_start:
        resp = client.post(
            "/api/training/start-video-training",
            json={
                "stream_url": "https://www.youtube.com/watch?v=mockLiveStream123",
                "epochs": 3,
                "max_frames": 30,
                "base_weights": "models/yolov8n.pt"
            },
            headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert mock_start.called


def test_apply_weights_endpoint(client, auth_headers):
    """Test hot-reloading weights via API."""
    # Applying non-existent file returns 400
    resp = client.post(
        "/api/training/apply-weights",
        json={"weights_path": "models/non_existent_weights_xyz.pt"},
        headers=auth_headers
    )
    assert resp.status_code == 400

    # Applying models/yolov8n.pt (which exists) succeeds
    if os.path.exists("models/yolov8n.pt"):
        resp2 = client.post(
            "/api/training/apply-weights",
            json={"weights_path": "models/yolov8n.pt"},
            headers=auth_headers
        )
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "success"


def test_resolve_stream_source():
    """Test resolver handling local file, YouTube URL, and generic URLs."""
    # Non-existent file / raw link returns original
    resolved = resolve_stream_source("https://my-rtsp-stream.local/live.m3u8")
    assert resolved == "https://my-rtsp-stream.local/live.m3u8"

    # YouTube URL mock test
    with patch("yt_dlp.YoutubeDL") as mock_ydl:
        instance = mock_ydl.return_value.__enter__.return_value
        instance.extract_info.return_value = {"url": "https://googlevideo.com/videoplayback?mock=1"}
        yt_resolved = resolve_stream_source("https://www.youtube.com/watch?v=liveTest")
        assert yt_resolved == "https://googlevideo.com/videoplayback?mock=1"
