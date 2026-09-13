import pytest
import numpy as np
import cv2
import yaml
import os

from ai_behavior.night_enhancement.adaptive_threshold import compute_tau_visibility
from ai_behavior.tamper.laplacian_check import check_laplacian_variance
from ai_behavior.tamper.histogram_check import HistogramChecker
from ai_behavior.orchestrator.pipeline import FrameOrchestrator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "thresholds.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

def test_visibility_threshold():
    # Create a dummy image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    tau = compute_tau_visibility(img)
    # Tau should be within configured limits
    limits = config.get("visibility_limits", [0.4, 0.8])
    assert limits[0] <= tau <= limits[1]

def test_laplacian_tamper_alert():
    # Create a completely blank image (variance 0)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    var, is_lens_covered = check_laplacian_variance(img)
    assert var < config.get("tamper_variance_collapse", 12.0)
    assert is_lens_covered is True

def test_histogram_blindfold():
    # Create an image that is almost completely one color
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    checker = HistogramChecker()
    histogram_flag, frame_delta = checker.check_histogram(img)
    assert histogram_flag is True

def mock_detection_stage(frame):
    # Mock person A's detection stage returning a dummy human
    return [
        {
            "track_id": "uuid-1234",
            "entity_type": "human",
            "bbox": [10, 10, 50, 50],
            "confidence": 0.95,
            "attributes": {
                "upper_color": "blue",
                "gender": "male"
            },
            "location": {"lat": 0.0, "lon": 0.0}
        }
    ]

def test_pipeline_orchestrator():
    orchestrator = FrameOrchestrator()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = orchestrator.process_frame(
        frame=img, 
        camera_id="CAM-01", 
        frame_ref="frame-1", 
        run_detection_stage_callback=mock_detection_stage
    )
    
    assert "frame_quality" in result
    assert "enhanced" in result["frame_quality"]
    assert "tau_visibility" in result["frame_quality"]
    assert "tamper_signal" in result["frame_quality"]
    
    entities = result["entities"]
    assert len(entities) == 1
    
    human = entities[0]
    assert "posture" in human["attributes"]
    assert "props" in human["attributes"]

    orchestrator.close()
