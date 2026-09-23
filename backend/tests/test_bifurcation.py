import pytest
from datetime import datetime, timezone
from backend.app.schemas import (
    FrameAnalysis, FrameQuality, TamperSignal, EntityDetection, EntityAttributes, LocationPoint
)
from backend.app.rule_engine.bifurcation import bifurcation_engine

@pytest.fixture
def sample_geofence():
    return [
        [78.1640, 29.9450],
        [78.1660, 29.9450],
        [78.1660, 29.9470],
        [78.1640, 29.9470],
        [78.1640, 29.9450]
    ]

def test_passive_logging_outside_geofence(sample_geofence):
    """Standard movement outside geofence must result in passive logging (FR-ALR-01)."""
    frame = FrameAnalysis(
        camera_id="CAM-01",
        timestamp=datetime.now(timezone.utc),
        frame_ref="frame_001.jpg",
        frame_quality=FrameQuality(tamper_signal=TamperSignal(is_tampered=False, laplacian_variance=500.0)),
        entities=[
            EntityDetection(
                track_id="t-1",
                entity_type="human",
                bbox=[100, 100, 200, 200],
                confidence=0.85,
                attributes=EntityAttributes(upper_color="blue", lower_color="black"),
                location=LocationPoint(lat=29.9500, lon=78.1700)  # Outside geofence
            )
        ]
    )

    decisions, camera_alert = bifurcation_engine.evaluate_frame(frame, sample_geofence)
    assert camera_alert is None
    assert len(decisions) == 1
    d = decisions[0]
    assert d.is_alert is False
    assert d.retention_tier == "passive"
    assert d.alert_type is None

def test_active_alert_geofence_breach(sample_geofence):
    """Crossing into geofence must trigger an active alert with protected retention (FR-ALR-02.1)."""
    frame = FrameAnalysis(
        camera_id="CAM-01",
        timestamp=datetime.now(timezone.utc),
        frame_ref="frame_002.jpg",
        frame_quality=FrameQuality(tamper_signal=TamperSignal(is_tampered=False, laplacian_variance=500.0)),
        entities=[
            EntityDetection(
                track_id="t-2",
                entity_type="human",
                bbox=[100, 100, 200, 200],
                confidence=0.88,
                attributes=EntityAttributes(upper_color="red", lower_color="blue"),
                location=LocationPoint(lat=29.9460, lon=78.1650)  # Inside geofence
            )
        ]
    )

    decisions, camera_alert = bifurcation_engine.evaluate_frame(frame, sample_geofence)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.is_alert is True
    assert d.alert_type == "geo_fence"
    assert d.retention_tier == "protected"

def test_multi_modal_corroboration(sample_geofence):
    """Breach combined with weapon must trigger correlated highest severity tier (FR-ALR-04)."""
    frame = FrameAnalysis(
        camera_id="CAM-01",
        timestamp=datetime.now(timezone.utc),
        frame_ref="frame_003.jpg",
        frame_quality=FrameQuality(tamper_signal=TamperSignal(is_tampered=False, laplacian_variance=500.0)),
        entities=[
            EntityDetection(
                track_id="t-3",
                entity_type="human",
                bbox=[100, 100, 200, 200],
                confidence=0.92,
                attributes=EntityAttributes(posture="crouching", props=["weapon"]),
                location=LocationPoint(lat=29.9460, lon=78.1650)  # Inside geofence
            )
        ]
    )

    decisions, camera_alert = bifurcation_engine.evaluate_frame(frame, sample_geofence)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.is_alert is True
    assert d.alert_type == "correlated"
    assert d.retention_tier == "protected"
    assert "weapon" in d.rule_fired

def test_camera_tamper_alert():
    """Laplacian variance below 100.0 must trigger tamper alert."""
    frame = FrameAnalysis(
        camera_id="CAM-02",
        timestamp=datetime.now(timezone.utc),
        frame_ref="frame_004.jpg",
        frame_quality=FrameQuality(tamper_signal=TamperSignal(is_tampered=True, laplacian_variance=45.0)),
        entities=[]
    )

    decisions, camera_alert = bifurcation_engine.evaluate_frame(frame, None)
    assert camera_alert is not None
    assert camera_alert["alert_type"] == "tamper"
    assert camera_alert["retention_tier"] == "protected"
