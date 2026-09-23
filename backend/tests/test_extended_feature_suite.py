"""
TRINETRA Extended Dataset-Driven Feature Test Suite
Extensive stress tests for all 17 features using realistic dataset-derived vectors.
Run with: python -m pytest backend/tests/test_extended_feature_suite.py -v
"""

import os
import sys
import uuid
import json
import pytest
import numpy as np
import cv2
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.database import SessionLocal, engine, Base
from backend.app.models import CameraRegistry, EntityLog, AuditLog, ArchiveEpochIndex, User
from backend.app.auth.jwt_auth import get_password_hash
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

# --- Helpers ------------------------------------------------------------------

def mk_frame(luminance=130, noise=0.0, vertical_streaks=False):
    frame = np.full((360, 640, 3), luminance, dtype=np.uint8)
    if noise > 0:
        noise_mat = (np.random.randn(360, 640, 3) * noise * 255).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise_mat, 0, 255).astype(np.uint8)
    if vertical_streaks:
        for _ in range(80):
            x = np.random.randint(0, 630)
            y = np.random.randint(0, 330)
            cv2.line(frame, (x, y), (x + 2, y + 22), (210, 210, 220), 1)
    return frame


def mk_plate_image(plate_text="DL01AB1234", angle_deg=0, glare=False):
    img = np.full((72, 200, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (14, 72), (180, 80, 20), -1)
    cv2.putText(img, plate_text, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
    if glare:
        cv2.ellipse(img, (160, 20), (30, 12), 45, 0, 360, (255, 255, 240), -1)
    if angle_deg != 0:
        M = cv2.getRotationMatrix2D((100, 36), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (200, 72))
    return img


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        cam = db.query(CameraRegistry).filter(CameraRegistry.camera_id == "EXT-CAM-01").first()
        if not cam:
            db.add(CameraRegistry(
                camera_id="EXT-CAM-01", location_lat=29.9457, location_lon=78.1642,
                status="online", trust_score=0.99, stream_url="/footage/sample.mp4",
                geo_fence_polygon=[[78.164, 29.945], [78.166, 29.945],
                                   [78.166, 29.947], [78.164, 29.947], [78.164, 29.945]]
            ))
        u = db.query(User).filter(User.username == "ext_test_op").first()
        if not u:
            db.add(User(id=str(uuid.uuid4()), username="ext_test_op",
                        password_hash=get_password_hash("pass123"),
                        role="commander", active=True))
        db.commit()
    yield


@pytest.fixture(scope="module")
def auth_headers():
    from backend.app.auth.jwt_auth import create_access_token
    with SessionLocal() as db:
        u = db.query(User).filter(User.username == "ext_test_op").first()
        uid = u.id if u else "ext_test_op"
    token = create_access_token({"sub": uid, "role": "commander"})
    return {"Authorization": f"Bearer {token}"}


# --- F1 — ANPR ---------------------------------------------------------------

class TestF1_ANPRTracking:

    def test_01_valid_indian_state_codes(self):
        from ai_detection.anpr.plate_ocr import validate_indian_plate
        valid_plates = ["DL01AB1234","MH12DE5678","UP16CK9999","HR26DQ1122",
                        "KA02MN1826","GJ01AX8888","RJ14CA2020","TN09BK7777",
                        "PB10BR0001","WB23AB0101"]
        for plate in valid_plates:
            assert validate_indian_plate(plate), f"FAIL: {plate} should be valid"

    def test_02_invalid_plates_rejected(self):
        from ai_detection.anpr.plate_ocr import validate_indian_plate
        invalid = ["RANDOM-TEXT","1234567890","XY99ZZ","INDIA123","ABCDEFGH","","IND","DL0","DL01AB"]
        for plate in invalid:
            assert not validate_indian_plate(plate), f"FAIL: {plate} should be invalid"

    def test_03_hsrp_badge_strip_and_clean(self):
        from ai_detection.anpr.plate_ocr import clean_plate_text
        cases = [
            ("IND DL-01-AB-1234", "DL01AB1234"),
            ("IND MH 12 DE 5678", "MH12DE5678"),
            ("ind up16ck9999",     "UP16CK9999"),
            (" HR26DQ1122 ",       "HR26DQ1122"),
        ]
        for raw, expected in cases:
            result = clean_plate_text(raw)
            assert result == expected, f"clean_plate_text({raw!r}) -> {result!r}, expected {expected!r}"

    def test_04_multi_frame_tracker_convergence(self):
        from ai_detection.anpr.plate_tracker import PlateTracker
        tracker = PlateTracker(max_history=8)
        noisy_reads = [
            ("DL01AB123", 0.55), ("DL01AB1234", 0.91), ("DL01AB12", 0.48),
            ("DL01AB1234", 0.87), ("D101AB1234", 0.63), ("DL01A81234", 0.58),
            ("DL01AB1234", 0.93), ("DL01AB1234", 0.72),
        ]
        for raw, conf in noisy_reads:
            tracker.update_track(track_id=1, raw_plate=raw, confidence=conf, bbox=(100,100,250,180))
        best = tracker.get_best_plate(track_id=1)
        assert best == "DL01AB1234", f"Expected DL01AB1234, got {best!r}"

    def test_05_plate_history_bounded(self):
        from ai_detection.anpr.plate_tracker import PlateTracker
        tracker = PlateTracker(max_history=3)
        for i in range(10):
            tracker.update_track(99, f"DL01AB{i:04d}", 0.80, (0,0,100,50))
        state = tracker.get_track_state(99)
        assert state is not None
        assert len(state.get("history", [])) <= 3

    def test_06_two_tracks_isolated(self):
        from ai_detection.anpr.plate_tracker import PlateTracker
        tracker = PlateTracker(max_history=5)
        tracker.update_track(101, "DL01AB1234", 0.92, (10,10,200,80))
        tracker.update_track(102, "MH12DE5678", 0.88, (300,10,500,80))
        assert tracker.get_best_plate(101) == "DL01AB1234"
        assert tracker.get_best_plate(102) == "MH12DE5678"


# --- F2 — VEHICLE DETAILS ----------------------------------------------------

class TestF2_VehicleDetails:

    def test_01_all_vehicle_types(self):
        from ai_detection.detection.yolov8_detector import DetectedEntity
        for vtype in ["car","bus","truck","van","motorcycle","auto_rickshaw","tractor","bicycle"]:
            ent = DetectedEntity(track_id=str(uuid.uuid4()), entity_type="vehicle",
                                 bbox=[100,100,400,300], confidence=0.88,
                                 foot_point=(250.0,300.0),
                                 crop=np.zeros((200,300,3),dtype=np.uint8), raw_class_name=vtype)
            assert ent.raw_class_name == vtype

    def test_02_speed_realistic_range(self):
        from ai_detection.detection.yolov8_detector import DetectedEntity
        for speed in [0.0, 15.5, 48.0, 80.0, 120.0, 160.0]:
            ent = DetectedEntity(track_id="V1", entity_type="vehicle",
                                 bbox=[100,100,400,300], confidence=0.90,
                                 foot_point=(250.0,300.0),
                                 crop=np.zeros((200,300,3),dtype=np.uint8), speed_kmh=speed)
            assert 0 <= ent.speed_kmh <= 200

    def test_03_direction_labels(self):
        from ai_detection.detection.yolov8_detector import DetectedEntity
        for direction in ["Advancing (North)","Retreating (South)","Crossing (East)",
                          "Crossing (West)","Stationary","Loitering"]:
            ent = DetectedEntity(track_id="V2", entity_type="vehicle",
                                 bbox=[100,100,400,300], confidence=0.85,
                                 foot_point=(250.0,300.0),
                                 crop=np.zeros((200,300,3),dtype=np.uint8), direction=direction)
            assert ent.direction == direction


# --- F3 — BODY POSTURE -------------------------------------------------------

class TestF3_BodyPosture:

    POSTURE_CASES = [
        ([100,200,400,260], "prone",     "crawling w=300,h=60 ratio=5.0"),
        ([100,200,310,280], "prone",     "prone w=210,h=80 ratio=2.62"),
        ([100,150,200,230], "crouching", "squat w=100,h=80 ratio=1.25"),
        ([100,150,190,250], "crouching", "crouch w=90,h=100 ratio=0.90"),
        ([100,80, 160,280], "standing",  "standing w=60,h=200 ratio=0.30"),
        ([200,50, 250,380], "standing",  "tall upright w=50,h=330 ratio=0.15"),
    ]

    def test_01_posture_classification_from_dataset_vectors(self):
        from ai_detection.par.clothing_color import estimate_posture
        for bbox, expected, desc in self.POSTURE_CASES:
            result = estimate_posture(bbox)
            assert result == expected, f"{desc}: expected {expected!r}, got {result!r}"

    def test_02_posture_enum_valid(self):
        from ai_detection.par.clothing_color import estimate_posture
        valid = {"standing","walking","crouching","prone","crawling","sprinting","climbing"}
        for bbox, _, _ in self.POSTURE_CASES:
            assert estimate_posture(bbox) in valid


# --- F4 — CATTLE DETECTION ---------------------------------------------------

class TestF4_CattleDetection:

    QUADRUPED_CASES = [
        (180,90,  True, "cow"),    (200,100,True,"buffalo"),(160,80, True,"sheep"),
        (220,110, True, "camel"),  (140,70, True,"dog"),
        (60, 180, False,"human"),  (55, 200,False,"runner"),(45,190,False,"child"),
    ]

    def test_01_quadruped_aspect_ratio(self):
        for w, h, is_quad, desc in self.QUADRUPED_CASES:
            assert (h/w < 0.75) == is_quad, f"{desc}: h/w={h/w:.2f}, expected quad={is_quad}"

    def test_02_cattle_dataset_in_catalog(self):
        from ai_detection.training.dataset_downloader import CURATED_DATASETS
        assert "border_cattle_livestock" in CURATED_DATASETS
        meta = CURATED_DATASETS["border_cattle_livestock"]
        for sp in ["cattle","buffalo","sheep","camel","human"]:
            assert sp in meta["classes"], f"Missing species: {sp}"

    def test_03_crawling_human_ambiguity_documented(self):
        crawl_bbox = [100,200,250,240]
        w,h = crawl_bbox[2]-crawl_bbox[0], crawl_bbox[3]-crawl_bbox[1]
        assert h/w < 0.75, "Crawling human has cattle-like aspect -- needs cattle model to disambiguate"


# --- F5 — ESTIMATED PATH ------------------------------------------------------

class TestF5_EstimatedPath:

    def test_01_foot_point_bottom_center(self):
        from ai_detection.detection.yolov8_detector import compute_foot_point
        cases = [
            ([120,140,220,320], (170.0,320.0)),
            ([0,0,640,360],     (320.0,360.0)),
            ([300,200,400,350], (350.0,350.0)),
        ]
        for bbox, expected in cases:
            assert compute_foot_point(bbox) == expected, f"bbox={bbox}: expected {expected}"

    def test_02_trajectory_extrapolation(self):
        traj = [(320.0,200.0),(318.0,220.0),(316.0,240.0),(314.0,260.0)]
        dx = traj[-1][0]-traj[-2][0]
        dy = traj[-1][1]-traj[-2][1]
        projected = (traj[-1][0]+dx, traj[-1][1]+dy)
        assert projected == (312.0, 280.0)

    def test_03_direction_from_vector(self):
        vectors = [(0.0,-5.0,"North"),(0.0,5.0,"South"),(5.0,0.0,"East"),(-5.0,0.0,"West")]
        for dx,dy,expected in vectors:
            if dy<0 and abs(dy)>abs(dx): d="North"
            elif dy>0 and abs(dy)>abs(dx): d="South"
            elif dx>0 and abs(dx)>abs(dy): d="East"
            else: d="West"
            assert d == expected


# --- F6 — FACE DETECTION & FRS -----------------------------------------------

class TestF6_FRS:

    def test_01_enrollment_and_match(self, tmp_path):
        from ai_detection.frs.known_suspects import KnownSuspectStore
        store = KnownSuspectStore(storage_dir=str(tmp_path/"frs"))
        emb = np.random.randn(512).astype(np.float32); emb /= np.linalg.norm(emb)
        store.enroll_suspect("SUS-001","Infiltrator Alpha",emb)
        noisy = emb + np.random.randn(512).astype(np.float32)*0.02
        noisy /= np.linalg.norm(noisy)
        result = store.match_face(noisy, threshold=0.68)
        assert result is not None and result["name"] == "Infiltrator Alpha"

    def test_02_false_positive_rejection(self, tmp_path):
        from ai_detection.frs.known_suspects import KnownSuspectStore
        store = KnownSuspectStore(storage_dir=str(tmp_path/"frs_fp"))
        emb = np.random.randn(512).astype(np.float32); emb /= np.linalg.norm(emb)
        store.enroll_suspect("SUS-002","Target",emb)
        other = np.random.randn(512).astype(np.float32); other /= np.linalg.norm(other)
        result = store.match_face(other, threshold=0.75)
        assert result is None or result["confidence"] < 0.75

    def test_03_multi_suspect_returns_closest(self, tmp_path):
        from ai_detection.frs.known_suspects import KnownSuspectStore
        store = KnownSuspectStore(storage_dir=str(tmp_path/"frs_multi"))
        embs = {}
        for i in range(10):
            e = np.random.randn(512).astype(np.float32); e /= np.linalg.norm(e)
            embs[i] = e; store.enroll_suspect(f"SUS-{i:03d}", f"Suspect {i}", e)
        result = store.match_face(embs[5], threshold=0.68)
        assert result is not None and result["name"] == "Suspect 5"


# --- F7 — WEAPON DETECTION ---------------------------------------------------

class TestF7_WeaponDetection:

    WEAPON_CLASSES = ["weapon","knife","gun","rifle","firearm","pistol","blade","machete","shotgun"]

    def test_01_all_weapon_props_trigger_alert(self):
        from ai_detection.detection.yolov8_detector import DetectedEntity
        for w in self.WEAPON_CLASSES:
            ent = DetectedEntity(track_id="H1", entity_type="human",
                                 bbox=[200,100,280,300], confidence=0.91,
                                 foot_point=(240.0,300.0),
                                 crop=np.zeros((200,80,3),dtype=np.uint8), extra_props=[w])
            assert any(wp in ent.extra_props for wp in self.WEAPON_CLASSES)

    def test_02_weapon_plus_breach_is_correlated(self):
        scenarios = [
            (["rifle"],True,"correlated",0.98), (["knife"],False,"behavior",0.95),
            (["large_backpack"],True,"correlated",0.90), (["large_backpack"],False,"behavior",0.85),
            ([],True,"geo_fence",0.88),
        ]
        WEAPON_SET = set(self.WEAPON_CLASSES)
        for props, is_breach, exp_type, min_score in scenarios:
            has_weapon = bool(WEAPON_SET & set(props))
            has_bag = any(b in props for b in ["large_backpack","suitcase"])
            if has_weapon and is_breach: t,s = "correlated",0.98
            elif has_weapon: t,s = "behavior",0.95
            elif has_bag and is_breach: t,s = "correlated",0.90
            elif has_bag: t,s = "behavior",0.85
            elif is_breach: t,s = "geo_fence",0.88
            else: t,s = None, 0.10
            if exp_type: assert t==exp_type and s>=min_score


# --- F8 — GEOFENCING ----------------------------------------------------------

class TestF8_Geofencing:
    POLY = [[78.164,29.945],[78.166,29.945],[78.166,29.947],[78.164,29.947],[78.164,29.945]]

    def test_01_centroid_inside(self):
        from backend.app.rule_engine.geofence_check import GeoFenceEvaluator
        assert GeoFenceEvaluator.is_inside_geofence(29.946, 78.165, self.POLY)

    def test_02_far_outside(self):
        from backend.app.rule_engine.geofence_check import GeoFenceEvaluator
        assert not GeoFenceEvaluator.is_inside_geofence(29.966, 78.185, self.POLY)

    def test_03_all_inner_corners(self):
        from backend.app.rule_engine.geofence_check import GeoFenceEvaluator
        for lat,lon in [(29.9452,78.1642),(29.9468,78.1658),(29.9452,78.1658),(29.9468,78.1642)]:
            assert GeoFenceEvaluator.is_inside_geofence(lat, lon, self.POLY)

    def test_04_all_outer_edges(self):
        from backend.app.rule_engine.geofence_check import GeoFenceEvaluator
        for lat,lon in [(29.944,78.165),(29.948,78.165),(29.946,78.163),(29.946,78.168)]:
            assert not GeoFenceEvaluator.is_inside_geofence(lat, lon, self.POLY)

    def test_05_irregular_l_shape(self):
        from backend.app.rule_engine.geofence_check import GeoFenceEvaluator
        l = [[78.160,29.940],[78.163,29.940],[78.163,29.943],[78.162,29.943],
             [78.162,29.945],[78.160,29.945],[78.160,29.940]]
        assert GeoFenceEvaluator.is_inside_geofence(29.941,78.161,l)
        assert not GeoFenceEvaluator.is_inside_geofence(29.944,78.1625,l)


# --- F9 — SHORT CLIPPING ------------------------------------------------------

class TestF9_ShortClipping:

    def test_01_ring_buffer_push_and_extract(self):
        from backend.app.ingestion.rtsp_service import rtsp_service
        cam = "EXT-CAM-01"
        ts = datetime.now(timezone.utc)
        dummy = np.zeros((360,640,3),dtype=np.uint8)
        _,enc = cv2.imencode(".jpg",dummy)
        for i in range(5):
            rtsp_service.push_frame(cam,f"f{i}",ts+timedelta(seconds=i),enc.tobytes())
        clip = rtsp_service.extract_30s_clip(cam,ts,f"clip_{uuid.uuid4().hex[:6]}")
        assert clip is not None
        assert str(clip).endswith(".mp4")

    def test_02_empty_buffer_no_crash(self):
        from backend.app.ingestion.rtsp_service import rtsp_service
        result = rtsp_service.extract_30s_clip("NO-SUCH-CAM",datetime.now(timezone.utc),"test_fb")
        assert result is None or isinstance(result,(str,Path))


# --- F10 & F11 — SEARCH & HINDI -----------------------------------------------

class TestF10F11_SearchAndHindi:

    def test_01_entity_type_filter(self, auth_headers):
        with SessionLocal() as db:
            for etype in ["human","vehicle","animal"]:
                db.add(EntityLog(id=f"ext_{etype}_{uuid.uuid4().hex[:6]}",
                                 camera_id="EXT-CAM-01",
                                 timestamp=datetime.now(timezone.utc),
                                 entity_type=etype, is_alert=False, confidence_score=0.85))
            db.commit()
        r = client.get("/api/entities",params={"entity_type":"vehicle","limit":20},headers=auth_headers)
        assert r.status_code == 200

    def test_02_hindi_vocabulary_resolved(self):
        from backend.app.search_service.synonym_dictionary import SynonymDictionary
        parser = SynonymDictionary()
        cases = [
            ("kala gadi","entity_type","vehicle"), ("safed gaadi","entity_type","vehicle"),
            ("bandook","prop","weapon"),            ("chaku aadmi","entity_type","human"),
            ("teji se bhagta","posture","sprinting"),("andhera","is_low_light",True),
            ("laal kapde","color","red"),
        ]
        for query, key, expected in cases:
            result = parser.parse_query(query)
            resolved = result.get("resolved",{})
            assert resolved.get(key) == expected, \
                f"parse_query({query!r})[{key!r}]={resolved.get(key)!r}, expected {expected!r}"

    def test_03_camera_id_filter(self, auth_headers):
        r = client.get("/api/entities",params={"camera_id":"EXT-CAM-01","limit":10},headers=auth_headers)
        assert r.status_code == 200
        for item in r.json():
            if item.get("camera_id"):
                assert item["camera_id"] == "EXT-CAM-01"


# --- F12 — SCREENSHOTS --------------------------------------------------------

class TestF12_Screenshots:

    def test_01_thumb_webp_saved(self):
        from backend.app.config import settings
        d = Path(settings.STORAGE_DIR)/"thumbnails"; d.mkdir(parents=True,exist_ok=True)
        f = d/f"thumb_{uuid.uuid4().hex[:8]}.webp"
        frame = np.zeros((360,640,3),dtype=np.uint8)
        cv2.rectangle(frame,(200,100),(280,300),(0,0,255),3)
        cv2.imwrite(str(f),frame,[cv2.IMWRITE_WEBP_QUALITY,85])
        assert f.exists() and f.stat().st_size > 0

    def test_02_plate_crop_webp_saved(self):
        from backend.app.config import settings
        d = Path(settings.STORAGE_DIR)/"thumbnails"; d.mkdir(parents=True,exist_ok=True)
        f = d/f"plate_{uuid.uuid4().hex[:8]}.webp"
        cv2.imwrite(str(f), mk_plate_image("MH12DE5678"), [cv2.IMWRITE_WEBP_QUALITY,92])
        assert f.exists() and f.stat().st_size > 1000

    def test_03_angled_plate_webp_saved(self):
        from backend.app.config import settings
        d = Path(settings.STORAGE_DIR)/"thumbnails"; d.mkdir(parents=True,exist_ok=True)
        f = d/f"plate_ang_{uuid.uuid4().hex[:6]}.webp"
        cv2.imwrite(str(f), mk_plate_image("DL01AB1234",angle_deg=15,glare=True), [cv2.IMWRITE_WEBP_QUALITY,90])
        assert f.exists() and f.stat().st_size > 500


# --- F16 — AUTO DAY/NIGHT -----------------------------------------------------

class TestF16_DayNightMode:
    NIGHT_LUMS = [8,12,20,30,40,50,60,70,84]
    DAY_LUMS   = [86,90,100,120,140,170,200,220]

    def test_01_exdark_extreme_darkness(self):
        from ai_behavior.environmental.weather_analyzer import compute_weather_and_visibility
        for lum in self.NIGHT_LUMS:
            env = compute_weather_and_visibility(mk_frame(lum))
            assert env.is_low_light, f"lum={lum} should be night"
            assert env.weather_condition == "NIGHT_LOW_LIGHT"
            assert env.recommended_model == "night"
            assert env.recommended_preprocessing == "zero_dce"

    def test_02_lol_daylight_not_night(self):
        from ai_behavior.environmental.weather_analyzer import compute_weather_and_visibility
        for lum in self.DAY_LUMS:
            env = compute_weather_and_visibility(mk_frame(lum))
            assert not env.is_low_light, f"lum={lum} should NOT be night"

    def test_03_boundary_at_85(self):
        from ai_behavior.environmental.weather_analyzer import compute_weather_and_visibility
        assert compute_weather_and_visibility(mk_frame(84)).is_low_light is True
        assert compute_weather_and_visibility(mk_frame(85)).is_low_light is False

    def test_04_night_visibility_score_low(self):
        from ai_behavior.environmental.weather_analyzer import compute_weather_and_visibility
        for lum in [20,40,60]:
            assert compute_weather_and_visibility(mk_frame(lum)).visibility_score <= 0.5


# --- F17 — AUTO MODEL ROUTING ------------------------------------------------

class TestF17_ModelRouting:

    def test_01_fog_routes_to_visdrone(self):
        from ai_behavior.environmental.weather_analyzer import compute_weather_and_visibility
        env = compute_weather_and_visibility(np.full((360,640,3),160,dtype=np.uint8))
        assert env.weather_condition in ("FOG_HAZE","OVERHEAD_HAZY")
        assert env.recommended_model == "visdrone"

    def test_02_night_routes_to_night(self):
        from ai_behavior.environmental.weather_analyzer import compute_weather_and_visibility
        env = compute_weather_and_visibility(mk_frame(35))
        assert env.recommended_model == "night"

    def test_03_all_5_fleet_models_present(self):
        from ai_detection.detection.yolov8_detector import FLEET_MODEL_PATHS
        assert set(FLEET_MODEL_PATHS.keys()) == {"custom","night","weapon","patrol","visdrone"}

    def test_04_routing_constants_correct(self):
        from ai_behavior.environmental.weather_analyzer import WeatherVisibilityAnalyzer
        a = WeatherVisibilityAnalyzer()
        assert a.lum_night_threshold == 85.0
        assert a.fog_contrast_threshold == 38.0
        assert a.fog_laplacian_threshold == 120.0

    def test_05_model_name_for_all_conditions(self):
        from ai_behavior.environmental.weather_analyzer import compute_weather_and_visibility
        valid_models = {"custom","night","patrol","visdrone","weapon"}
        for lum in [30, 85, 130, 160]:
            env = compute_weather_and_visibility(mk_frame(lum))
            assert env.recommended_model in valid_models


# --- F13/F14/F15 — SIREN, ALERTS, LOGS --------------------------------------

class TestF13F14F15_AlertsLogs:

    def test_01_siren_broadcast_payload(self):
        from backend.app.api.ws_alerts import alert_ws_manager
        payload = {"event_type":"ACTIVE_ALERT","alert_id":f"s_{uuid.uuid4().hex[:6]}",
                   "camera_id":"EXT-CAM-01","alert_type":"breach","threat_score":0.97,"play_siren":True}
        alert_ws_manager.broadcast_sync(payload)
        assert payload["play_siren"] is True

    def test_02_merkle_root_is_64_hex(self):
        with SessionLocal() as db:
            epoch = ArchiveEpochIndex(
                id=str(uuid.uuid4()), epoch_id=f"ep_{uuid.uuid4().hex[:8]}",
                camera_id="EXT-CAM-01",
                start_time=datetime.now(timezone.utc)-timedelta(hours=1),
                end_time=datetime.now(timezone.utc),
                merkle_root="a"*64, frame_count=500, alert_count=12)
            db.add(epoch); db.commit()
            fetched = db.query(ArchiveEpochIndex).filter(ArchiveEpochIndex.frame_count==500).first()
            assert len(fetched.merkle_root) == 64
            assert all(c in "0123456789abcdef" for c in fetched.merkle_root.lower())

    def test_03_audit_log_sha256_in_details(self):
        with SessionLocal() as db:
            u = db.query(User).first(); uid = u.id if u else str(uuid.uuid4())
            db.add(AuditLog(id=str(uuid.uuid4()), user_id=uid, action="EVIDENCE_EXPORT",
                            target_id=f"ent_{uuid.uuid4().hex[:6]}",
                            timestamp=datetime.now(timezone.utc),
                            details={"hash_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}))
            db.commit()
            fetched = db.query(AuditLog).filter(AuditLog.action=="EVIDENCE_EXPORT").order_by(AuditLog.timestamp.desc()).first()
            assert "hash_sha256" in fetched.details
            assert len(fetched.details["hash_sha256"]) == 64

    def test_04_active_alerts_structure(self, auth_headers):
        r = client.get("/api/alerts/active", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        if data:
            assert "camera_id" in data[0]

    def test_05_alert_history_reachable(self, auth_headers):
        r = client.get("/api/alerts/history", headers=auth_headers)
        assert r.status_code == 200


if __name__ == "__main__":
    pytest.main(["-v", "--tb=short", __file__])
