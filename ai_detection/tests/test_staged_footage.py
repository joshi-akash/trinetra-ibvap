"""
Smoke test runner against test_footage/ for TRINETRA AI Detection.

Runs the detection and recognition pipeline against video clips in test_footage/
(or synthetic staged test frames if video files are pending staging).
"""

from __future__ import annotations

import glob
import os
import sys
import unittest
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_detection import run_detection_stage
from ai_detection.frs.known_suspects import KnownSuspectStore


class TestStagedFootageSmoke(unittest.TestCase):
    """Smoke test runner for staged video footage."""

    def setUp(self):
        self.footage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../test_footage"))
        self.calib_data = {
            "camera_id": "CAM-01",
            "reference_points": [
                {"image_pt": [100, 900], "world_pt": [0, 0]},
                {"image_pt": [1800, 900], "world_pt": [50, 0]},
                {"image_pt": [1800, 500], "world_pt": [50, 80]},
                {"image_pt": [100, 500], "world_pt": [0, 80]},
            ],
            "vertical_scale_cm_per_pixel": 0.5,
        }

    def test_synthetic_daylight_frame_pipeline(self):
        """Smoke test full pipeline on a 1080p synthetic daylight frame."""
        frame = np.full((1080, 1920, 3), 150, dtype=np.uint8)
        entities = run_detection_stage(
            frame=frame,
            camera_id="CAM-01",
            calibration_data=self.calib_data,
            is_low_light=False,
        )
        self.assertIsInstance(entities, list)

    def test_synthetic_low_light_frame_pipeline(self):
        """Smoke test full pipeline on a dark 1080p synthetic frame."""
        frame = np.full((1080, 1920, 3), 20, dtype=np.uint8)
        entities = run_detection_stage(
            frame=frame,
            camera_id="CAM-02",
            calibration_data=self.calib_data,
            is_low_light=True,
        )
        self.assertIsInstance(entities, list)

    def test_real_video_clips_if_available(self):
        """If any .mp4 clips exist in test_footage/, process the first 10 frames of each."""
        video_files = glob.glob(os.path.join(self.footage_dir, "*.mp4"))
        if not video_files:
            # Staged clips not present yet — skip gracefully
            return

        try:
            import cv2
            for v_path in video_files:
                cap = cv2.VideoCapture(v_path)
                frame_count = 0
                while cap.isOpened() and frame_count < 10:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    entities = run_detection_stage(frame=frame, camera_id="CAM-TEST")
                    self.assertIsInstance(entities, list)
                    frame_count += 1
                cap.release()
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
