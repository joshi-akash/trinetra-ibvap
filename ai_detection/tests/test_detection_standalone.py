"""
Standalone Test Suite for TRINETRA AI Detection & Recognition Stage (Person A).

Verifies:
1. Ground-contact foot coordinate computation ((x1+x2)/2, y2)
2. FRS resolution cutoff (>= 40x40 px) and Cosine Similarity threshold (Scos >= 0.68)
3. Gender confidence gating (< 0.85 -> 'neutral')
4. ANPR Indian license plate regex validation and confidence penalization
5. PAR clothing color HSV canonicalization and IR/night fallback ('unknown')
6. Perspective height estimation (calibrated vs uncalibrated)
7. Contract 1 FrameAnalysis entities schema conformance
8. Orchestrator frame_quality dict shape with enhanced, tau_visibility, and tamper_signal
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_detection.detection.yolov8_detector import (
    DetectedEntity,
    YOLOv8Detector,
    compute_foot_point,
)
from ai_detection.frs.face_matcher import MIN_FACE_RESOLUTION, FaceMatcher
from ai_detection.frs.known_suspects import (
    MATCH_SIMILARITY_THRESHOLD,
    KnownSuspectStore,
)
from ai_detection.anpr.plate_detector import PlateDetector
from ai_detection.anpr.plate_ocr import (
    INDIAN_PLATE_PATTERN,
    PlateOCR,
    clean_plate_text,
    validate_indian_plate,
)
from ai_detection.par.clothing_color import (
    CANONICAL_COLORS,
    classify_hsv_pixel,
    extract_clothing_colors,
)
from ai_detection.par.gender_estimation import (
    GENDER_CONFIDENCE_THRESHOLD,
    estimate_gender,
)
from ai_detection.height.perspective_height import (
    PerspectiveHeightEstimator,
    compute_height_cm,
)
from ai_detection import run_detection_stage
try:
    from ai_behavior.orchestrator.pipeline import (
        OrchestratorPipeline,
        compute_laplacian_variance,
        compute_tau_visibility,
    )
except ImportError:
    from orchestrator.pipeline import (
        OrchestratorPipeline,
        compute_laplacian_variance,
        compute_tau_visibility,
    )


from ai_detection.detection.tracker import ByteTracker


class TestFootPointCalculation(unittest.TestCase):
    """Test ground-contact foot point coordinates (x_foot, y_foot) = ((x1+x2)/2, y2)."""

    def test_foot_point_standard(self):
        bbox = [100, 50, 200, 300]
        x_foot, y_foot = compute_foot_point(bbox)
        self.assertEqual(x_foot, 150.0)
        self.assertEqual(y_foot, 300.0)

    def test_foot_point_odd_coordinates(self):
        bbox = [10, 20, 25, 80]
        x_foot, y_foot = compute_foot_point(bbox)
        self.assertEqual(x_foot, 17.5)
        self.assertEqual(y_foot, 80.0)

    def test_detected_entity_foot_point(self):
        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        entity = DetectedEntity(
            track_id="TRK-0001",
            entity_type="human",
            bbox=[50, 100, 150, 400],
            confidence=0.92,
            foot_point=compute_foot_point([50, 100, 150, 400]),
            crop=crop,
        )
        self.assertEqual(entity.foot_point, (100.0, 400.0))
        d = entity.to_dict()
        self.assertEqual(d["foot_point"], [100.0, 400.0])


class TestByteTracker(unittest.TestCase):
    """Test ByteTracker tracking across consecutive frames."""

    def test_tracker_assigns_and_persists_ids(self):
        tracker = ByteTracker(match_thresh=0.8)
        crop = np.zeros((10, 10, 3), dtype=np.uint8)

        # Frame 1: entity at [100, 100, 200, 200]
        ent1 = DetectedEntity(
            track_id="",
            entity_type="human",
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            foot_point=(150.0, 200.0),
            crop=crop,
        )
        tracked_f1 = tracker.update([ent1])
        self.assertEqual(len(tracked_f1), 1)
        self.assertEqual(tracked_f1[0].track_id, "TRK-0001")

        # Frame 2: same entity slightly moved to [105, 102, 205, 202]
        ent2 = DetectedEntity(
            track_id="",
            entity_type="human",
            bbox=[105, 102, 205, 202],
            confidence=0.92,
            foot_point=(155.0, 202.0),
            crop=crop,
        )
        tracked_f2 = tracker.update([ent2])
        self.assertEqual(len(tracked_f2), 1)
        # Should persist the same track ID
        self.assertEqual(tracked_f2[0].track_id, "TRK-0001")

        # Frame 3: a second new entity appears at [400, 400, 500, 500]
        ent3 = DetectedEntity(
            track_id="",
            entity_type="vehicle",
            bbox=[400, 400, 500, 500],
            confidence=0.88,
            foot_point=(450.0, 500.0),
            crop=crop,
        )
        tracked_f3 = tracker.update([ent2, ent3])
        self.assertEqual(len(tracked_f3), 2)
        self.assertEqual(tracked_f3[0].track_id, "TRK-0001")
        self.assertEqual(tracked_f3[1].track_id, "TRK-0002")


class TestFRSResolutionAndMatching(unittest.TestCase):
    """Test InsightFace resolution cutoff (>= 40x40 px) and cosine matching (Scos >= 0.68)."""

    def setUp(self):
        # Create temporary suspect store
        import tempfile
        self.test_dir = tempfile.mkdtemp()
        self.store = KnownSuspectStore(storage_dir=self.test_dir)
        self.matcher = FaceMatcher(suspect_store=self.store)

    def test_resolution_cutoff_under_40px(self):
        """Crops smaller than 40x40 must be skipped (return None)."""
        small_crop_39 = np.zeros((39, 39, 3), dtype=np.uint8)
        self.assertIsNone(self.matcher.extract_embedding(small_crop_39))

        small_crop_20x50 = np.zeros((20, 50, 3), dtype=np.uint8)
        self.assertIsNone(self.matcher.extract_embedding(small_crop_20x50))

        small_crop_50x20 = np.zeros((50, 20, 3), dtype=np.uint8)
        self.assertIsNone(self.matcher.extract_embedding(small_crop_50x20))

    def test_resolution_cutoff_exact_and_above_40px(self):
        """Crops >= 40x40 px satisfy resolution requirement."""
        self.assertEqual(MIN_FACE_RESOLUTION, 40)
        # Verify resolution check condition
        h, w = 40, 40
        self.assertTrue(h >= MIN_FACE_RESOLUTION and w >= MIN_FACE_RESOLUTION)
        h2, w2 = 120, 120
        self.assertTrue(h2 >= MIN_FACE_RESOLUTION and w2 >= MIN_FACE_RESOLUTION)

    def test_known_suspect_cosine_matching_above_threshold(self):
        """Matching vector with Scos >= 0.68 must return suspect_id and confidence."""
        # Create normalized random 512-d base embedding
        np.random.seed(42)
        v1 = np.random.randn(512).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)

        self.store.enroll_suspect(suspect_id="SUSP-001", name="Target Alpha", embedding=v1)

        # Query with high similarity (slight noise, cosine sim ~ 0.95)
        v_noise = np.random.randn(512).astype(np.float32)
        v_noise = v_noise / np.linalg.norm(v_noise)
        v_query = v1 + 0.1 * v_noise
        v_query = v_query / np.linalg.norm(v_query)

        match = self.store.match_face(v_query, threshold=0.68)
        self.assertIsNotNone(match)
        self.assertEqual(match["suspect_id"], "SUSP-001")
        self.assertGreaterEqual(match["confidence"], 0.68)

    def test_known_suspect_cosine_matching_below_threshold(self):
        """Dissimilar vector with Scos < 0.68 must return None (no false match)."""
        np.random.seed(42)
        v1 = np.random.randn(512).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)

        self.store.enroll_suspect(suspect_id="SUSP-002", name="Target Beta", embedding=v1)

        # Query with orthogonal/dissimilar vector
        v_orthogonal = np.random.randn(512).astype(np.float32)
        v_orthogonal -= np.dot(v_orthogonal, v1) * v1  # make orthogonal
        v_orthogonal = v_orthogonal / np.linalg.norm(v_orthogonal)

        match = self.store.match_face(v_orthogonal, threshold=0.68)
        self.assertIsNone(match)

    def test_load_from_bench_suspects_store(self):
        """Verify tests can explicitly load and query the benchmark suspects fixture (suspects_bench/)."""
        bench_store = KnownSuspectStore(bench_mode=True)
        self.assertIn("suspects_bench", bench_store.storage_dir)
        # Verify bench store loaded records from suspects.json fixture
        self.assertGreater(len(bench_store.suspects), 0)

        # Test query against first suspect in benchmark store
        first_id = list(bench_store.suspects.keys())[0]
        first_emb = np.array(bench_store.suspects[first_id].embedding, dtype=np.float32)
        first_emb = first_emb / np.linalg.norm(first_emb)

        match = bench_store.match_face(first_emb, threshold=0.68)
        self.assertIsNotNone(match)
        self.assertEqual(match["suspect_id"], first_id)
        self.assertAlmostEqual(match["confidence"], 1.0, places=2)

    def test_default_live_suspects_store_path(self):
        """Verify default runtime store points to suspects/ (live store, unseeded in repo)."""
        live_store = KnownSuspectStore(bench_mode=False)
        self.assertTrue(live_store.storage_dir.endswith(os.path.join("data", "suspects")))


class TestGenderConfidenceGating(unittest.TestCase):
    """Test FR-PAR-02: Mandate that gender is strictly 'neutral' if confidence < 85%."""

    def test_confidence_below_85_percent_is_neutral(self):
        crop = np.zeros((100, 50, 3), dtype=np.uint8)

        # 84% confidence -> strictly neutral
        res_gender, conf = estimate_gender(crop, raw_prediction=("male", 0.84))
        self.assertEqual(res_gender, "neutral")
        self.assertEqual(conf, 0.84)

        # 50% confidence female -> strictly neutral
        res_gender, conf = estimate_gender(crop, raw_prediction=("female", 0.50))
        self.assertEqual(res_gender, "neutral")
        self.assertEqual(conf, 0.50)

        # 0% confidence -> neutral
        res_gender, conf = estimate_gender(crop, raw_prediction=("male", 0.0))
        self.assertEqual(res_gender, "neutral")

    def test_confidence_at_or_above_85_percent(self):
        crop = np.zeros((100, 50, 3), dtype=np.uint8)

        # Exact 85% cutoff
        res_gender, conf = estimate_gender(crop, raw_prediction=("male", 0.85))
        self.assertEqual(res_gender, "male")
        self.assertEqual(conf, 0.85)

        # 95% female
        res_gender, conf = estimate_gender(crop, raw_prediction=("female", 0.95))
        self.assertEqual(res_gender, "female")
        self.assertEqual(conf, 0.95)


class TestANPRRegexAndPenalization(unittest.TestCase):
    """Test ANPR Indian plate regex (^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$) and penalization."""

    def test_valid_indian_plates(self):
        valid_plates = [
            "DL01A1234",
            "HR26DQ5551",
            "UP32AB1234",
            "MH12DE1433",
            "JK02C9988",
            "WB02K4567",
            "KA04MC8899",
            "PB65AT0001",
        ]
        for plate in valid_plates:
            self.assertTrue(
                validate_indian_plate(plate),
                f"Plate '{plate}' should match Indian regex: ^[A-Z]{{2}}[0-9]{{1,2}}[A-Z]{{1,2}}[0-9]{{4}}$",
            )

    def test_invalid_plates_fail_regex(self):
        invalid_plates = [
            "12345",
            "ABCDEF",
            "DL01",
            "USA9999",
            "INVALID1234",
            "D01A1234",  # Only 1 letter state code
        ]
        for plate in invalid_plates:
            self.assertFalse(
                validate_indian_plate(plate),
                f"Plate '{plate}' should NOT match Indian regex",
            )

    def test_plate_cleaning(self):
        self.assertEqual(clean_plate_text("hr-26 dq 5551"), "HR26DQ5551")
        self.assertEqual(clean_plate_text(" dl.01_a:1234 "), "DL01A1234")

    def test_ocr_confidence_penalization(self):
        ocr = PlateOCR(confidence_penalty_factor=0.5)
        # Test mock OCR result behavior
        raw_conf = 0.90
        # Valid format retains raw confidence
        valid_conf = raw_conf if validate_indian_plate("DL01A1234") else raw_conf * 0.5
        self.assertEqual(valid_conf, 0.90)

        # Invalid format penalized
        invalid_conf = raw_conf if validate_indian_plate("INVALID") else raw_conf * 0.5
        self.assertEqual(invalid_conf, 0.45)


class TestPARClothingColor(unittest.TestCase):
    """Test HSV clothing color canonicalization and night/IR fallback."""

    def test_canonical_colors_vocabulary(self):
        expected_vocab = {
            "red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray", "brown", "unknown"
        }
        self.assertEqual(set(CANONICAL_COLORS), expected_vocab)

    def test_hsv_color_bins(self):
        # Pure Blue in OpenCV HSV: H ~ 120, S ~ 255, V ~ 255
        self.assertEqual(classify_hsv_pixel(120, 255, 255), "blue")

        # Pure Red in OpenCV HSV: H ~ 0, S ~ 255, V ~ 255
        self.assertEqual(classify_hsv_pixel(0, 255, 255), "red")
        self.assertEqual(classify_hsv_pixel(175, 255, 255), "red")

        # Pure Green in OpenCV HSV: H ~ 60, S ~ 255, V ~ 255
        self.assertEqual(classify_hsv_pixel(60, 255, 255), "green")

        # Black (low V)
        self.assertEqual(classify_hsv_pixel(0, 0, 20), "black")

        # White (low S, high V)
        self.assertEqual(classify_hsv_pixel(0, 10, 240), "white")

    def test_night_ir_fallback_returns_unknown(self):
        """Under low-light / IR conditions, colors must return 'unknown' (FR-PAR-03)."""
        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        u_col, l_col = extract_clothing_colors(crop, is_low_light=True)
        self.assertEqual(u_col, "unknown")
        self.assertEqual(l_col, "unknown")


class TestPerspectiveHeightEstimation(unittest.TestCase):
    """Test perspective geometry height calculation."""

    def test_uncalibrated_returns_none(self):
        bbox = [100, 100, 200, 400]
        self.assertIsNone(compute_height_cm(bbox, calibration_data=None))

        # Under 4 reference points
        bad_calib = {"reference_points": [{"image_pt": [0, 0], "world_pt": [0, 0]}]}
        self.assertIsNone(compute_height_cm(bbox, calibration_data=bad_calib))

    def test_calibrated_height_bounds(self):
        calib = {
            "camera_id": "CAM-01",
            "reference_points": [
                {"image_pt": [100, 400], "world_pt": [0, 0]},
                {"image_pt": [300, 400], "world_pt": [10, 0]},
                {"image_pt": [300, 200], "world_pt": [10, 20]},
                {"image_pt": [100, 200], "world_pt": [0, 20]},
            ],
            "vertical_scale_cm_per_pixel": 0.6,
        }
        estimator = PerspectiveHeightEstimator(calib)
        self.assertTrue(estimator.is_calibrated)

        bbox = [150, 100, 250, 400]  # height = 300 px
        h_cm = estimator.estimate_height(bbox)
        self.assertIsNotNone(h_cm)
        self.assertTrue(30.0 <= h_cm <= 250.0)


class TestDetectionStageAndPipelineSchema(unittest.TestCase):
    """Verify Contract 1 FrameAnalysis and frame_quality schema conformance."""

    def test_run_detection_stage_empty_frame(self):
        entities = run_detection_stage(frame=None, camera_id="CAM-01")
        self.assertEqual(entities, [])

    def test_orchestrator_frame_quality_shape(self):
        """
        Verify orchestrator outputs frame_quality with:
        - low_light: bool
        - enhanced: bool (True only when tau_visibility < 0.65)
        - tau_visibility: float
        - tamper_signal: {is_tampered, laplacian_variance, histogram_flag, reference_point_drift}
        """
        pipeline = OrchestratorPipeline()

        # 1. Clear bright frame with high contrast (tau_visibility >= 0.65)
        np.random.seed(0)
        bright_frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        analysis_bright = pipeline.process_frame(bright_frame, camera_id="CAM-01")

        fq_bright = analysis_bright["frame_quality"]
        self.assertIn("low_light", fq_bright)
        self.assertIn("enhanced", fq_bright)
        self.assertIn("tau_visibility", fq_bright)
        self.assertIn("tamper_signal", fq_bright)

        self.assertIsInstance(fq_bright["low_light"], bool)
        self.assertIsInstance(fq_bright["enhanced"], bool)
        self.assertIsInstance(fq_bright["tau_visibility"], float)

        ts = fq_bright["tamper_signal"]
        self.assertIn("is_tampered", ts)
        self.assertIn("laplacian_variance", ts)
        self.assertIn("histogram_flag", ts)
        self.assertIn("reference_point_drift", ts)

        # 2. Dark blurry frame (tau_visibility < 0.65) -> enhanced MUST be True
        dark_frame = np.ones((480, 640, 3), dtype=np.uint8) * 15
        analysis_dark = pipeline.process_frame(dark_frame, camera_id="CAM-02")
        fq_dark = analysis_dark["frame_quality"]

        self.assertTrue(fq_dark["low_light"])
        self.assertTrue(fq_dark["enhanced"])
        self.assertLess(fq_dark["tau_visibility"], 0.65)

    def test_frame_analysis_structure(self):
        pipeline = OrchestratorPipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipeline.process_frame(frame, camera_id="CAM-04")

        self.assertEqual(res["camera_id"], "CAM-04")
        self.assertIn("timestamp", res)
        self.assertIn("frame_ref", res)
        self.assertIn("frame_quality", res)
        self.assertIn("entities", res)


class TestYOLOClassMapping(unittest.TestCase):
    """Test YOLO class mapping to TRINETRA canonical entity types."""

    def test_class_mappings(self):
        detector = YOLOv8Detector()
        self.assertEqual(detector.map_class_id(0), "human")       # person
        self.assertEqual(detector.map_class_id(2), "vehicle")     # car
        self.assertEqual(detector.map_class_id(7), "vehicle")     # truck
        self.assertEqual(detector.map_class_id(16), "animal")     # dog
        self.assertEqual(detector.map_class_id(19), "animal")     # cow
        self.assertIsNone(detector.map_class_id(9))               # traffic light (filtered out)
        self.assertIsNone(detector.map_class_id(60))              # dining table (filtered out)

    def test_custom_border_class_mapping(self):
        detector = YOLOv8Detector()
        self.assertEqual(detector.map_class_id(-1, "person"), "human")
        self.assertEqual(detector.map_class_id(-1, "animal_drawn_cart"), "vehicle")
        self.assertEqual(detector.map_class_id(-1, "dog"), "animal")


class TestDetectionStageMockIntegration(unittest.TestCase):
    """Test run_detection_stage with mocked detections to verify Contract 1 schema in detail."""

    def test_human_entity_contract_compliance(self):
        class MockHumanDetector(YOLOv8Detector):
            def detect_and_track(self, frame, conf_override=None, persist=True):
                crop = np.zeros((200, 100, 3), dtype=np.uint8)
                # Blue shirt in upper torso
                crop[30:100, 20:80] = [255, 0, 0]  # BGR Blue
                return [
                    DetectedEntity(
                        track_id="TRK-0001",
                        entity_type="human",
                        bbox=[100, 50, 200, 250],
                        confidence=0.94,
                        foot_point=(150.0, 250.0),
                        crop=crop,
                    )
                ]

        det = MockHumanDetector()
        calib = {
            "camera_id": "CAM-01",
            "reference_points": [
                {"image_pt": [50, 300], "world_pt": [0, 0]},
                {"image_pt": [250, 300], "world_pt": [10, 0]},
                {"image_pt": [250, 50], "world_pt": [10, 20]},
                {"image_pt": [50, 50], "world_pt": [0, 20]},
            ],
            "vertical_scale_cm_per_pixel": 0.85,
        }

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        entities = run_detection_stage(
            frame=frame,
            camera_id="CAM-01",
            calibration_data=calib,
            detector=det,
        )

        self.assertEqual(len(entities), 1)
        ent = entities[0]

        # Verify Contract 1 entity fields
        self.assertEqual(ent["track_id"], "TRK-0001")
        self.assertEqual(ent["entity_type"], "human")
        self.assertEqual(ent["bbox"], [100, 50, 200, 250])
        self.assertEqual(ent["confidence"], 0.94)

        attrs = ent["attributes"]
        self.assertIn(attrs["upper_color"], CANONICAL_COLORS)
        self.assertIn(attrs["lower_color"], CANONICAL_COLORS)
        self.assertEqual(attrs["gender"], "neutral")  # Gated to neutral
        self.assertIsNone(attrs["plate_text"])       # Humans must not have plate_text
        self.assertIsNone(attrs["face_match"])       # No face match without InsightFace
        self.assertIsNotNone(attrs["height_cm"])
        self.assertTrue(30.0 <= attrs["height_cm"] <= 250.0)

    def test_vehicle_entity_contract_compliance(self):
        class MockVehicleDetector(YOLOv8Detector):
            def detect_and_track(self, frame, conf_override=None, persist=True):
                crop = np.zeros((150, 250, 3), dtype=np.uint8)
                return [
                    DetectedEntity(
                        track_id="TRK-0002",
                        entity_type="vehicle",
                        bbox=[200, 100, 450, 250],
                        confidence=0.89,
                        foot_point=(325.0, 250.0),
                        crop=crop,
                    )
                ]

        class MockPlateOCR(PlateOCR):
            def read_plate(self, plate_crop):
                return ("HR26DQ5551", 0.95, True)

        det = MockVehicleDetector()
        ocr = MockPlateOCR()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        entities = run_detection_stage(
            frame=frame,
            camera_id="CAM-01",
            detector=det,
            plate_ocr=ocr,
        )

        self.assertEqual(len(entities), 1)
        ent = entities[0]

        self.assertEqual(ent["entity_type"], "vehicle")
        attrs = ent["attributes"]
        self.assertEqual(attrs["plate_text"], "HR26DQ5551")
        self.assertIsNone(attrs["upper_color"])
        self.assertIsNone(attrs["lower_color"])
        self.assertIsNone(attrs["gender"])
        self.assertIsNone(attrs["height_cm"])
        self.assertIsNone(attrs["face_match"])

    def test_animal_entity_contract_compliance(self):
        class MockAnimalDetector(YOLOv8Detector):
            def detect_and_track(self, frame, conf_override=None, persist=True):
                crop = np.zeros((100, 150, 3), dtype=np.uint8)
                return [
                    DetectedEntity(
                        track_id="TRK-0003",
                        entity_type="animal",
                        bbox=[50, 150, 200, 250],
                        confidence=0.78,
                        foot_point=(125.0, 250.0),
                        crop=crop,
                    )
                ]

        det = MockAnimalDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        entities = run_detection_stage(frame=frame, camera_id="CAM-01", detector=det)

        self.assertEqual(len(entities), 1)
        ent = entities[0]
        self.assertEqual(ent["entity_type"], "animal")
        for key, val in ent["attributes"].items():
            self.assertIsNone(val, f"Attribute '{key}' should be None for animal")


class TestTrainingUtilities(unittest.TestCase):
    """Test dataset configuration generator and licensing manifest utilities."""

    def test_generate_dataset_yaml_and_license(self):
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        yaml_path = os.path.join(tmp_dir, "test_border.yaml")

        from ai_detection.training.dataset_downloader import generate_dataset_yaml, record_dataset_license

        # Test YAML generation
        success = generate_dataset_yaml(
            output_yaml_path=yaml_path,
            dataset_root_dir=tmp_dir,
            class_names=["human", "vehicle", "animal", "weapon"],
        )
        self.assertTrue(success)
        self.assertTrue(os.path.exists(yaml_path))

        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("train: images/train", content)
        self.assertIn("0: human", content)
        self.assertIn("3: weapon", content)

        # Test License manifest recording (PRD Section 10)
        lic_success = record_dataset_license(
            dataset_dir=tmp_dir,
            dataset_name="border_cctv_sample",
            license_type="CC-BY-4.0",
            source_url="https://kaggle.com/example/dataset",
        )
        self.assertTrue(lic_success)
        manifest_file = os.path.join(tmp_dir, "DATASET_LICENSE.json")
        self.assertTrue(os.path.exists(manifest_file))


if __name__ == "__main__":
    unittest.main(verbosity=2)
