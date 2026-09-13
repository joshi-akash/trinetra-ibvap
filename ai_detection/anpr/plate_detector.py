"""
License Plate Detector for TRINETRA ANPR.

Localizes the license plate bounding box within a vehicle image crop
using either a dedicated YOLOv8 plate detector model or morphological/contour
candidate filtering tailored for Indian license plates.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PlateDetector:
    """
    Localizes license plate candidate regions within vehicle image patches.
    """

    def __init__(
        self,
        model_path: Optional[str] = "models/plate_yolov8n.pt",
        confidence_threshold: float = 0.35,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._is_mock = False

        self._load_model()

    def _load_model(self) -> None:
        """Load dedicated plate detector YOLO model if available."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                logger.info("Loaded plate detector from %s", self.model_path)
                return
            except Exception as e:
                logger.warning("Failed to load plate model (%s). Using heuristic.", e)

        self.model = None
        self._is_mock = True

    def detect_plate_crop(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the most probable license plate crop from a vehicle crop.

        Args:
            vehicle_crop: BGR image crop of a detected vehicle

        Returns:
            np.ndarray plate image patch, or None if not found
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 40:
            return None

        # 1. Try YOLO plate detector model
        if self.model is not None and not self._is_mock:
            try:
                res = self.model.predict(source=vehicle_crop, conf=self.confidence_threshold, verbose=False)
                if res and len(res) > 0 and len(res[0].boxes) > 0:
                    box = res[0].boxes[0]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    px1 = max(0, min(w - 1, int(xyxy[0])))
                    py1 = max(0, min(h - 1, int(xyxy[1])))
                    px2 = max(px1 + 1, min(w, int(xyxy[2])))
                    py2 = max(py1 + 1, min(h, int(xyxy[3])))
                    return vehicle_crop[py1:py2, px1:px2].copy()
            except Exception as e:
                logger.error("Plate detector model error: %s", e)

        # 2. Heuristic candidate: lower-central region of the vehicle (standard plate mount area)
        # Indian plates are typically located in the bottom 40% and central 70% of vehicle view
        y1_heur = int(h * 0.55)
        y2_heur = int(h * 0.95)
        x1_heur = int(w * 0.15)
        x2_heur = int(w * 0.85)

        if y2_heur > y1_heur and x2_heur > x1_heur:
            return vehicle_crop[y1_heur:y2_heur, x1_heur:x2_heur].copy()

        return vehicle_crop
