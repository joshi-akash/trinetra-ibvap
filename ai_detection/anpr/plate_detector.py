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

    def detect_plate_candidate_crops(self, vehicle_crop: np.ndarray) -> List[np.ndarray]:
        """
        Extract multiple candidate license plate crops from a vehicle crop across
        lower bumper, front grille, and full crop scales.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 30:
            return [vehicle_crop]

        candidates = []

        # 1. Dedicated YOLO model prediction
        if self.model is not None and not self._is_mock:
            try:
                res = self.model.predict(source=vehicle_crop, conf=self.confidence_threshold, verbose=False)
                if res and len(res) > 0 and len(res[0].boxes) > 0:
                    for box in res[0].boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        px1 = max(0, min(w - 1, int(xyxy[0])))
                        py1 = max(0, min(h - 1, int(xyxy[1])))
                        px2 = max(px1 + 1, min(w, int(xyxy[2])))
                        py2 = max(py1 + 1, min(h, int(xyxy[3])))
                        if (py2 - py1) >= 12 and (px2 - px1) >= 25:
                            candidates.append(vehicle_crop[py1:py2, px1:px2].copy())
            except Exception as e:
                logger.error("Plate detector model error: %s", e)

        # 2. Lower-central region (standard lower bumper mount)
        y1_heur = int(h * 0.50)
        y2_heur = int(h * 0.98)
        x1_heur = int(w * 0.10)
        x2_heur = int(w * 0.90)
        if y2_heur > y1_heur and x2_heur > x1_heur:
            candidates.append(vehicle_crop[y1_heur:y2_heur, x1_heur:x2_heur].copy())

        # 3. Upper-central region (grille / bonnet mount for SUVs, trucks, and high-clearance vehicles)
        y1_grille = int(h * 0.25)
        y2_grille = int(h * 0.75)
        x1_grille = int(w * 0.10)
        x2_grille = int(w * 0.90)
        if y2_grille > y1_grille and x2_grille > x1_grille:
            candidates.append(vehicle_crop[y1_grille:y2_grille, x1_grille:x2_grille].copy())

        # 4. Full vehicle crop
        candidates.append(vehicle_crop)
        return candidates

    def detect_plate_crop(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the primary license plate crop from a vehicle crop.
        """
        candidates = self.detect_plate_candidate_crops(vehicle_crop)
        return candidates[0] if candidates else None

