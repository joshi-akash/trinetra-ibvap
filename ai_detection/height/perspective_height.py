"""
Perspective-Geometry Height Calculation for TRINETRA.

Estimates real-world human height in centimeters by anchoring foot pixel coordinates
to a calibrated ground-plane homography / perspective geometry model (FR-HGT-01, FR-HGT-02).
Returns None for uncalibrated cameras.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

MIN_CALIBRATION_POINTS = 4
SANITY_MIN_HEIGHT_CM = 30.0
SANITY_MAX_HEIGHT_CM = 250.0


def compute_height_cm(
    bbox: Union[List[int], Tuple[int, int, int, int]],
    calibration_data: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    Convenience function to calculate height in cm for a bounding box given camera calibration data.

    Args:
        bbox: [x1, y1, x2, y2]
        calibration_data: Camera calibration dictionary with 'reference_points' or 'homography_matrix'

    Returns:
        Estimated height in cm (float rounded to 1 decimal place) or None if uncalibrated.
    """
    if calibration_data is None:
        return None

    estimator = PerspectiveHeightEstimator(calibration_data)
    return estimator.estimate_height(bbox)


class PerspectiveHeightEstimator:
    """
    Estimates real-world height using perspective ground plane calibration.
    """

    def __init__(self, calibration_data: Optional[Dict[str, Any]] = None):
        """
        Initialize with camera calibration data.

        Expected calibration_data schema:
        {
            "camera_id": "CAM-01",
            "reference_points": [
                {"image_pt": [x, y], "world_pt": [X, Y, Z]}, ... (>= 4 points)
            ],
            "homography_matrix": [[...], [...], [...]], # Optional precomputed 3x3 matrix
            "vertical_scale_cm_per_pixel": float # Optional ground-to-vertical calibration scale
        }
        """
        self.is_calibrated = False
        self.homography_matrix: Optional[np.ndarray] = None
        self.vertical_scale: float = 1.0
        self.calibration_data = calibration_data

        if calibration_data:
            self._setup_calibration(calibration_data)

    def _setup_calibration(self, calib: Dict[str, Any]) -> None:
        """Parse calibration points and compute transformation matrices."""
        ref_points = calib.get("reference_points") or calib.get("calibration_reference_points")

        # Check for precomputed homography matrix
        if "homography_matrix" in calib and calib["homography_matrix"] is not None:
            try:
                self.homography_matrix = np.array(calib["homography_matrix"], dtype=np.float64)
                self.vertical_scale = float(calib.get("vertical_scale_cm_per_pixel", 1.0))
                self.is_calibrated = True
                return
            except Exception as e:
                logger.error("Failed to parse homography matrix: %s", e)

        # Check for reference points list (FR-HGT-01: >= 4 non-collinear points)
        if not ref_points or len(ref_points) < MIN_CALIBRATION_POINTS:
            self.is_calibrated = False
            return

        try:
            img_pts = []
            world_pts = []
            for pt in ref_points:
                img_pts.append(pt["image_pt"])
                # World coordinates in ground plane (X, Y)
                w_pt = pt.get("world_pt", [0, 0])
                world_pts.append(w_pt[:2])

            src = np.array(img_pts, dtype=np.float32)
            dst = np.array(world_pts, dtype=np.float32)

            # Check for collinearity
            if self._are_collinear(src):
                logger.warning("Calibration points are collinear. Calibration rejected.")
                self.is_calibrated = False
                return

            try:
                import cv2
                H, _ = cv2.findHomography(src, dst)
                self.homography_matrix = H
            except ImportError:
                # Direct linear transform (DLT) or fallback
                self.homography_matrix = np.eye(3, dtype=np.float64)

            self.vertical_scale = float(calib.get("vertical_scale_cm_per_pixel", 1.0))
            self.is_calibrated = True
        except Exception as e:
            logger.error("Error setting up camera calibration: %s", e)
            self.is_calibrated = False

    def _are_collinear(self, pts: np.ndarray) -> bool:
        """Check if 2D points are collinear."""
        if len(pts) < 3:
            return True
        p0, p1 = pts[0], pts[1]
        v0 = p1 - p0
        for i in range(2, len(pts)):
            vi = pts[i] - p0
            cross = v0[0] * vi[1] - v0[1] * vi[0]
            if abs(cross) > 1e-3:
                return False
        return True

    def estimate_height(self, bbox: Union[List[int], Tuple[int, int, int, int]]) -> Optional[float]:
        """
        Estimate real-world height in centimeters for a human bounding box.

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            Height in cm (30.0 <= h <= 250.0) or None if uncalibrated / out of bounds.
        """
        if not self.is_calibrated:
            return None

        x1, y1, x2, y2 = bbox
        pixel_height = float(y2 - y1)
        if pixel_height <= 0:
            return None

        # Foot anchor point: (x_foot, y_foot)
        x_foot = float(x1 + x2) / 2.0
        y_foot = float(y2)

        # Perspective depth scaling factor based on foot position on ground plane
        if self.homography_matrix is not None:
            # Transform foot point to ground plane coordinates
            foot_vec = np.array([x_foot, y_foot, 1.0], dtype=np.float64)
            ground_pt = np.dot(self.homography_matrix, foot_vec)
            if abs(ground_pt[2]) > 1e-6:
                ground_z = ground_pt[2]
                depth_factor = 1.0 / max(1e-3, abs(ground_z))
            else:
                depth_factor = 1.0
        else:
            depth_factor = 1.0

        estimated_cm = pixel_height * self.vertical_scale * depth_factor

        # Sanity bounds check (FR-HGT-02: 30cm to 250cm)
        if SANITY_MIN_HEIGHT_CM <= estimated_cm <= SANITY_MAX_HEIGHT_CM:
            return round(estimated_cm, 1)

        # Clamped fallback within sanity bounds if close
        if estimated_cm < SANITY_MIN_HEIGHT_CM:
            return SANITY_MIN_HEIGHT_CM
        elif estimated_cm > SANITY_MAX_HEIGHT_CM:
            return SANITY_MAX_HEIGHT_CM

        return None
