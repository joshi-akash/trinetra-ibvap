"""
YOLOv8 Object Detector for TRINETRA AI Detection.

Loads YOLOv8 models (default: models/yolov8s.pt), performs bounding box detection,
filters classes strictly to 'human', 'vehicle', and 'animal', and computes
ground-contact foot coordinates (x_foot, y_foot) = ((x1 + x2) / 2, y2).
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# COCO Class ID mappings to TRINETRA canonical entity types
COCO_HUMAN_IDS = {0}  # person
COCO_VEHICLE_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
COCO_ANIMAL_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

# Fine-tuned border custom class mapping
BORDER_CLASS_MAP = {
    "person": "human",
    "human": "human",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "vehicle": "vehicle",
    "animal_drawn_cart": "vehicle",
    "animal": "animal",
    "dog": "animal",
    "cow": "animal",
    "horse": "animal",
    "sheep": "animal",
    "weapon": "weapon",
    "large_backpack": "large_backpack",
}


@dataclass
class DetectedEntity:
    """Represents an intermediate detected entity before full attribute extraction."""
    track_id: str
    entity_type: str  # 'human', 'vehicle', 'animal'
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    foot_point: Tuple[float, float]  # (x_foot, y_foot) = ((x1 + x2)/2, y2)
    crop: np.ndarray  # Cropped image patch BGR
    raw_class_name: Optional[str] = None
    direction: Optional[str] = None
    speed_kmh: Optional[float] = None
    posture: Optional[str] = None
    extra_props: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to basic dictionary structure."""
        return {
            "track_id": self.track_id,
            "entity_type": self.entity_type,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "foot_point": [round(self.foot_point[0], 2), round(self.foot_point[1], 2)],
        }


def compute_foot_point(bbox: Union[List[int], Tuple[int, int, int, int]]) -> Tuple[float, float]:
    """
    Compute ground-contact foot point coordinates.
    Formula: (x_foot, y_foot) = ((x1 + x2) / 2.0, y2)
    """
    x1, y1, x2, y2 = bbox
    x_foot = float(x1 + x2) / 2.0
    y_foot = float(y2)
    return (x_foot, y_foot)


class YOLOv8Detector:
    """
    YOLOv8 Object Detector for bbox inference and foot-point computation.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8s.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the YOLOv8 detector.

        Args:
            model_path: Path to YOLOv8 weights (e.g. models/yolov8s.pt)
            confidence_threshold: Minimum confidence score for detections
            device: 'cuda', 'cpu', or None for auto-detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._is_mock = False

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load Ultralytics YOLO model; fallback gracefully if not installed/found."""
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info("Loaded YOLOv8 model from %s", self.model_path)
            else:
                logger.warning(
                    "Model weights not found at %s. Initializing YOLO('yolov8s.pt')",
                    self.model_path,
                )
                self.model = YOLO("yolov8s.pt")
            if self.device:
                self.model.to(self.device)
        except Exception as e:
            logger.warning("Ultralytics YOLO unavailable or failed to load (%s). Operating in fallback mode.", e)
            self.model = None
            self._is_mock = True

    def map_class_id(self, class_id: int, class_name: Optional[str] = None) -> Optional[str]:
        """
        Map raw YOLO class ID or name to TRINETRA canonical entity_type:
        'human', 'vehicle', or 'animal'.
        """
        if class_name:
            cname = class_name.lower().strip()
            if cname in BORDER_CLASS_MAP:
                return BORDER_CLASS_MAP[cname]

        if class_id in COCO_HUMAN_IDS:
            return "human"
        elif class_id in COCO_VEHICLE_IDS:
            return "vehicle"
        elif class_id in COCO_ANIMAL_IDS:
            return "animal"
        return None

    def detect(
        self,
        frame: np.ndarray,
        conf_override: Optional[float] = None,
    ) -> List[DetectedEntity]:
        """
        Run object detection inference on a single frame.

        Args:
            frame: Input BGR image (np.ndarray)
            conf_override: Optional override for confidence threshold

        Returns:
            List of DetectedEntity objects with bounding boxes and foot points.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        conf = conf_override if conf_override is not None else self.confidence_threshold
        results: List[DetectedEntity] = []

        if self.model is not None and not self._is_mock:
            try:
                pred_results = self.model.predict(
                    source=frame,
                    conf=conf,
                    verbose=False,
                )

                for r in pred_results:
                    boxes = r.boxes
                    if boxes is None:
                        continue

                    for i in range(len(boxes)):
                        box = boxes[i]
                        cls_id = int(box.cls[0].item())
                        cls_name = r.names.get(cls_id, "") if hasattr(r, "names") else ""
                        entity_type = self.map_class_id(cls_id, cls_name)

                        # Filter strictly to human, vehicle, animal
                        if entity_type not in ("human", "vehicle", "animal"):
                            continue

                        score = float(box.conf[0].item())
                        if score < conf:
                            continue

                        # Temporary ID before tracking pass
                        t_id = f"TMP-{uuid.uuid4().hex[:8]}"

                        # Coordinates
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1 = max(0, min(w - 1, int(xyxy[0])))
                        y1 = max(0, min(h - 1, int(xyxy[1])))
                        x2 = max(x1 + 1, min(w, int(xyxy[2])))
                        y2 = max(y1 + 1, min(h, int(xyxy[3])))

                        bbox = [x1, y1, x2, y2]
                        foot_pt = compute_foot_point(bbox)
                        crop = frame[y1:y2, x1:x2].copy()

                        results.append(
                            DetectedEntity(
                                track_id=t_id,
                                entity_type=entity_type,
                                bbox=bbox,
                                confidence=score,
                                foot_point=foot_pt,
                                crop=crop,
                                raw_class_name=cls_name,
                            )
                        )
                return results
            except Exception as e:
                logger.error("Error during YOLOv8 detection inference: %s. Falling back.", e)

        return results

    def detect_and_track(
        self,
        frame: np.ndarray,
        conf_override: Optional[float] = None,
        persist: bool = True,
    ) -> List[DetectedEntity]:
        """
        Convenience wrapper detecting objects and associating tracks via ByteTracker.
        """
        from .tracker import ByteTracker
        if not hasattr(self, "_tracker") or self._tracker is None:
            self._tracker = ByteTracker()

        raw_detections = self.detect(frame, conf_override=conf_override)
        return self._tracker.update(raw_detections, frame=frame)
