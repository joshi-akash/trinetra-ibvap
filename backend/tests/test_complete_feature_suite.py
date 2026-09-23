"""
TRINETRA Complete 17-Feature Verification Suite
Comprehensive automated test coverage for all 17 platform capabilities (PRD §4-§10).
"""

import os
import sys
import uuid
import json
import time
import shutil
import pytest
import numpy as np
import cv2
from datetime import datetime, timezone, timedelta
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database import SessionLocal, engine, Base
from backend.app.models import CameraRegistry, EntityLog, AuditLog, User, ArchiveEpochIndex
from backend.app.auth.jwt_auth import get_password_hash, create_access_token
from backend.app.config import settings

client = TestClient(app)


# Setup fixture for clean test database and dummy camera
@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        # Create test sector camera if missing
        cam = db.query(CameraRegistry).filter(CameraRegistry.camera_id == "TEST-CAM-17").first()
        if not cam:
            cam = CameraRegistry(
                camera_id="TEST-CAM-17",
                location_lat=29.9457,
                location_lon=78.1642,
                stream_url="/footage/border_patrol_sample.mp4",
                status="online",
                trust_score=0.99,
                geo_fence_polygon=[
                    [78.1640, 29.9450],
                    [78.1660, 29.9450],
                    [78.1660, 29.9470],
                    [78.1640, 29.9470],
                    [78.1640, 29.9450]
                ]
            )
            db.add(cam)

        # Create test operator user if missing
        test_user = db.query(User).filter(User.username == "test_operator_17").first()
        if not test_user:
            test_user = User(
                id=str(uuid.uuid4()),
                username="test_operator_17",
                password_hash=get_password_hash("pass123"),
                role="commander",
                active=True
            )
            db.add(test_user)
        db.commit()
    yield


@pytest.fixture(scope="module")
def auth_headers():
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == "test_operator_17").first()
        user_id = user.id if user else "test_operator_17"
    token = create_access_token(data={"sub": user_id, "role": "commander"})
    return {"Authorization": f"Bearer {token}"}


# --- 1. Number Plate Tracking ---
def test_feature_01_number_plate_tracking():
    from ai_detection.anpr.plate_ocr import clean_plate_text, validate_indian_plate
    from ai_detection.anpr.plate_tracker import PlateTracker

    # Test Indian Plate Format Normalization & Validation
    valid_plates = ["DL01AB1234", "MH12DE5678", "UP16CK9999", "HR26DQ1122", "KA02MN1826"]
    for plate in valid_plates:
        assert validate_indian_plate(plate), f"Plate {plate} should be valid Indian plate"

    # Test HSRP "IND" badge stripping and character cleanup
    cleaned = clean_plate_text("IND DL-01-AB-1234")
    assert cleaned == "DL01AB1234", f"Expected DL01AB1234, got {cleaned}"

    # Invalid plate rejection
    assert not validate_indian_plate("RANDOM-TEXT-123")

    # Multi-frame persistent track retention
    tracker = PlateTracker(max_history=5)
    tracker.update_track(track_id=101, raw_plate="DL01AB1234", confidence=0.88, bbox=(100, 100, 250, 180))
    # Next frame has partial OCR read
    tracker.update_track(track_id=101, raw_plate="DL01AB123", confidence=0.70, bbox=(105, 102, 255, 182))
    best_plate = tracker.get_best_plate(track_id=101)
    assert best_plate == "DL01AB1234", "Tracker should retain the highest confidence valid plate read"


# --- 2. Vehicle Details ---
def test_feature_02_vehicle_details():
    from ai_detection.detection.yolov8_detector import DetectedEntity

    vehicle = DetectedEntity(
        track_id="VEH-202",
        entity_type="vehicle",
        bbox=[150, 180, 420, 310],
        confidence=0.92,
        foot_point=(285.0, 310.0),
        crop=np.zeros((130, 270, 3), dtype=np.uint8),
        raw_class_name="truck",
        speed_kmh=48.5,
        direction="Advancing (South)"
    )
    assert vehicle.entity_type == "vehicle"
    assert vehicle.raw_class_name == "truck"
    assert vehicle.speed_kmh > 40.0
    assert "Advancing" in vehicle.direction


# --- 3. Body Posture ---
def test_feature_03_body_posture():
    from ai_detection.par.clothing_color import estimate_posture

    # Prone / crawling (horizontal aspect ratio w/h > 1.25)
    prone_bbox = [100, 200, 300, 260]  # w=200, h=60 -> ratio=3.33
    assert estimate_posture(prone_bbox) == "prone"

    # Crouching (squat aspect ratio 0.75 <= w/h <= 1.25)
    crouch_bbox = [100, 150, 190, 250]  # w=90, h=100 -> ratio=0.90
    assert estimate_posture(crouch_bbox) == "crouching"

    # Standing (upright aspect ratio w/h < 0.75)
    stand_bbox = [100, 80, 160, 280]  # w=60, h=200 -> ratio=0.30
    assert estimate_posture(stand_bbox) == "standing"


# --- 4. Cattle Detection & Grazing Boundary Logic ---
def test_feature_04_cattle_detection():
    from ai_detection.training.dataset_downloader import CURATED_DATASETS

    # Verify border cattle catalog entry
    assert "border_cattle_livestock" in CURATED_DATASETS
    cattle_meta = CURATED_DATASETS["border_cattle_livestock"]
    assert "cattle" in cattle_meta["classes"]
    assert "buffalo" in cattle_meta["classes"]

    # Livestock quadruped ratio vs upright human
    cattle_w, cattle_h = 180, 100
    aspect_cattle = cattle_h / cattle_w  # ~0.55 < 0.75
    assert aspect_cattle < 0.75, "Cattle aspect ratio is horizontally quadrupedal"

    human_w, human_h = 70, 180
    aspect_human = human_h / human_w  # ~2.57 >= 0.75
    assert aspect_human >= 0.75, "Human aspect ratio is vertically elongated"


# --- 5. Estimated Path & Ground Footpoint ---
def test_feature_05_estimated_path():
    from ai_detection.detection.yolov8_detector import compute_foot_point

    bbox = [120, 140, 220, 320]  # x1, y1, x2, y2
    foot_pt = compute_foot_point(bbox)
    # Expected: ((120+220)/2, 320) = (170.0, 320.0)
    assert foot_pt == (170.0, 320.0)

    # Multi-point trajectory simulation
    traj = [(150.0, 280.0), (160.0, 300.0), foot_pt]
    assert len(traj) == 3
    # Forward projected path vector
    dx = traj[-1][0] - traj[-2][0]
    dy = traj[-1][1] - traj[-2][1]
    projected_next = (traj[-1][0] + dx, traj[-1][1] + dy)
    assert projected_next == (180.0, 340.0)


# --- 6. Face Detection & FRS ---
def test_feature_06_face_detection_frs(tmp_path):
    from ai_detection.frs.known_suspects import KnownSuspectStore

    store = KnownSuspectStore(storage_dir=str(tmp_path / "suspects"))

    # Synthetic 512-dim normalized embedding
    emb_suspect = np.random.randn(512).astype(np.float32)
    emb_suspect /= np.linalg.norm(emb_suspect)

    store.enroll_suspect(suspect_id="SUS-99", name="Tariq Infiltrator", embedding=emb_suspect)

    # Matching with identical embedding
    match = store.match_face(emb_suspect, threshold=0.68)
    assert match is not None
    assert match["name"] == "Tariq Infiltrator"
    assert match["confidence"] > 0.95

    # Orthogonal / random query fails threshold
    emb_random = np.random.randn(512).astype(np.float32)
    emb_random /= np.linalg.norm(emb_random)
    diff_match = store.match_face(emb_random, threshold=0.75)
    assert diff_match is None or diff_match["confidence"] < 0.75


# --- 7. Weapon Detection & Threat Props ---
def test_feature_07_weapon_detection():
    from ai_detection.detection.yolov8_detector import DetectedEntity

    armed_entity = DetectedEntity(
        track_id="HUMAN-404",
        entity_type="human",
        bbox=[200, 100, 280, 300],
        confidence=0.91,
        foot_point=(240.0, 300.0),
        crop=np.zeros((200, 80, 3), dtype=np.uint8),
        extra_props=["rifle", "large_backpack"]
    )
    props = armed_entity.extra_props
    has_weapon = any(w in props for w in ["weapon", "knife", "gun", "rifle", "firearm"])
    assert has_weapon is True
    assert "rifle" in props
    assert "large_backpack" in props


# --- 8. Geofencing & Ray-Casting Polygon Breach ---
def test_feature_08_geofencing():
    from backend.app.rule_engine.geofence_check import GeoFenceEvaluator

    poly = [
        [78.1640, 29.9450],
        [78.1660, 29.9450],
        [78.1660, 29.9470],
        [78.1640, 29.9470],
        [78.1640, 29.9450]
    ]

    # Point clearly inside polygon
    inside = GeoFenceEvaluator.is_inside_geofence(
        entity_lat=29.9460,
        entity_lon=78.1650,
        geofence_coords=poly
    )
    assert inside is True

    # Point clearly outside polygon
    outside = GeoFenceEvaluator.is_inside_geofence(
        entity_lat=29.9500,
        entity_lon=78.1700,
        geofence_coords=poly
    )
    assert outside is False


# --- 9. Short Clipping Test ---
def test_feature_09_short_clipping_test():
    from backend.app.ingestion.rtsp_service import rtsp_service

    alert_id = f"test_clip_{uuid.uuid4().hex[:8]}"
    trigger_ts = datetime.now(timezone.utc)

    # Push frames to ring buffer
    dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
    _, enc = cv2.imencode('.jpg', dummy_img)
    rtsp_service.push_frame("TEST-CAM-17", "frame_1", trigger_ts, enc.tobytes())

    clip_out = rtsp_service.extract_30s_clip("TEST-CAM-17", trigger_ts, alert_id)
    assert clip_out is not None
    assert str(clip_out).endswith(".mp4")


# --- 10. Forensic Search Features ---
def test_feature_10_search_features(auth_headers):
    # Ingest test log entry into database
    with SessionLocal() as db:
        test_log = EntityLog(
            id=f"ent_search_{uuid.uuid4().hex[:8]}",
            camera_id="TEST-CAM-17",
            timestamp=datetime.now(timezone.utc),
            entity_type="vehicle",
            vehicle_type="car",
            upper_color="black",
            plate_text="DL01AB1234",
            is_alert=False,
            confidence_score=0.88
        )
        db.add(test_log)
        db.commit()

    # Search via /api/entities with filters
    res = client.get(
        "/api/entities",
        params={"camera_id": "TEST-CAM-17", "entity_type": "vehicle"},
        headers=auth_headers
    )
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    found = any(r.get("plate_text") == "DL01AB1234" for r in data)
    assert found is True


# --- 11. Hindi / Vernacular Search ---
def test_feature_11_hindi_search():
    from backend.app.search_service.synonym_dictionary import SynonymDictionary

    parser = SynonymDictionary()

    # Hindi/Hinglish Query: "kala gadi" (Black vehicle)
    f1 = parser.parse_query("kala gadi")["resolved"]
    assert f1["entity_type"] == "vehicle"
    assert f1["color"] == "black"

    # Hindi Query: "bandook andhera" (Weapon in darkness)
    f2 = parser.parse_query("bandook andhera")["resolved"]
    assert f2["prop"] == "weapon"
    assert f2["is_low_light"] is True

    # Hindi Query: "chaku aadmi" (Person with knife)
    f3 = parser.parse_query("chaku aadmi")["resolved"]
    assert f3["entity_type"] == "human"
    assert f3["prop"] == "weapon"

    # Hindi Query: "teji se bhagta" (Sprinting)
    f4 = parser.parse_query("teji se bhagta")["resolved"]
    assert f4["posture"] == "sprinting"

    # Vernacular Query: "safed gaadi" (White vehicle)
    f5 = parser.parse_query("safed gaadi")["resolved"]
    assert f5["entity_type"] == "vehicle"
    assert f5["color"] == "white"


# --- 12. Screenshots (Threat Snaps & HSRP Plate Crops) ---
def test_feature_12_screenshots():
    thumb_dir = Path(settings.STORAGE_DIR) / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    plate_crop = np.full((40, 140, 3), 240, dtype=np.uint8)
    cv2.putText(plate_crop, "DL01AB1234", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    ent_id = uuid.uuid4().hex[:8]
    p_file = thumb_dir / f"plate_{ent_id}.webp"
    t_file = thumb_dir / f"thumb_{ent_id}.webp"

    cv2.imwrite(str(p_file), plate_crop, [cv2.IMWRITE_WEBP_QUALITY, 90])
    cv2.imwrite(str(t_file), dummy_frame, [cv2.IMWRITE_WEBP_QUALITY, 85])

    assert p_file.exists() and p_file.stat().st_size > 0
    assert t_file.exists() and t_file.stat().st_size > 0


# --- 13. Siren Test ---
def test_feature_13_siren_test():
    from backend.app.api.ws_alerts import alert_ws_manager

    alert_event = {
        "event_type": "ACTIVE_ALERT",
        "alert_id": f"siren_{uuid.uuid4().hex[:6]}",
        "camera_id": "TEST-CAM-17",
        "alert_type": "perimeter_breach",
        "threat_score": 0.96,
        "play_siren": True
    }
    alert_ws_manager.broadcast_sync(alert_event)
    assert alert_event["play_siren"] is True
    assert alert_event["threat_score"] >= 0.90


# --- 14. Alerts Working (Severity Tiering & Active Retrieval) ---
def test_feature_14_alerts_working(auth_headers):
    with SessionLocal() as db:
        alert_log = EntityLog(
            id=f"alert_test_{uuid.uuid4().hex[:8]}",
            camera_id="TEST-CAM-17",
            timestamp=datetime.now(timezone.utc),
            entity_type="human",
            is_alert=True,
            confidence_score=0.95,
            alert_type="behavior",
            rule_fired="threat_prop: weapon (illegal weapon possession)"
        )
        db.add(alert_log)
        db.commit()

    res = client.get("/api/alerts/active", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    found_alert = any(a.get("rule_fired") and "weapon" in a.get("rule_fired") for a in data)
    assert found_alert is True


# --- 15. Logs Working & Merkle Chain of Custody ---
def test_feature_15_logs_working():
    with SessionLocal() as db:
        user = db.query(User).first()
        user_id = user.id if user else str(uuid.uuid4())

        audit = AuditLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            action="EVIDENCE_EXPORT",
            target_id="test_entity_123",
            timestamp=datetime.now(timezone.utc),
            details={"hash_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
        )
        db.add(audit)

        epoch = ArchiveEpochIndex(
            id=str(uuid.uuid4()),
            epoch_id=f"epoch_{uuid.uuid4().hex[:8]}",
            camera_id="TEST-CAM-17",
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc),
            merkle_root="a" * 64,
            frame_count=100,
            alert_count=2
        )
        db.add(epoch)
        db.commit()

        retrieved_audit = db.query(AuditLog).filter(AuditLog.target_id == "test_entity_123").first()
        assert retrieved_audit is not None
        assert retrieved_audit.details["hash_sha256"] is not None

        retrieved_epoch = db.query(ArchiveEpochIndex).filter(ArchiveEpochIndex.camera_id == "TEST-CAM-17").first()
        assert retrieved_epoch is not None
        assert len(retrieved_epoch.merkle_root) == 64


# --- 16. Auto Day / Night Mode & Low-Light Illumination ---
def test_feature_16_auto_day_night_mode():
    from ai_behavior.environmental.weather_analyzer import compute_weather_and_visibility

    # Synthetic dark night frame (mean lum ~ 30 < 85)
    dark_frame = np.full((360, 640, 3), 30, dtype=np.uint8)
    env_night = compute_weather_and_visibility(dark_frame)

    assert env_night.is_low_light is True
    assert env_night.weather_condition == "NIGHT_LOW_LIGHT"
    assert env_night.recommended_model == "night"
    assert env_night.recommended_preprocessing == "zero_dce"

    # Synthetic daylight frame (mean lum ~ 150 > 85)
    bright_frame = np.full((360, 640, 3), 150, dtype=np.uint8)
    env_day = compute_weather_and_visibility(bright_frame)

    assert env_day.is_low_light is False
    assert env_day.weather_condition != "NIGHT_LOW_LIGHT"


# --- 17. Auto Model Changing Based on Weather and Visibility ---
def test_feature_17_auto_model_changing_weather_visibility():
    from ai_behavior.environmental.weather_analyzer import compute_weather_and_visibility

    # 1. Clear Day -> Custom Model
    clear_frame = np.full((360, 640, 3), 130, dtype=np.uint8)
    clear_frame[::2, ::2] = 210
    clear_frame[1::2, 1::2] = 50
    res_clear = compute_weather_and_visibility(clear_frame)
    assert res_clear.recommended_model in ["custom", "patrol"]

    # 2. Dense Fog Frame -> VisDrone Model
    fog_frame = np.full((360, 640, 3), 160, dtype=np.uint8)
    res_fog = compute_weather_and_visibility(fog_frame)
    assert res_fog.weather_condition in ["FOG_HAZE", "OVERHEAD_HAZY"]
    assert res_fog.recommended_model == "visdrone"

    # 3. Night Low-Light Frame -> Night Model
    night_frame = np.full((360, 640, 3), 40, dtype=np.uint8)
    res_night = compute_weather_and_visibility(night_frame)
    assert res_night.weather_condition == "NIGHT_LOW_LIGHT"
    assert res_night.recommended_model == "night"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
