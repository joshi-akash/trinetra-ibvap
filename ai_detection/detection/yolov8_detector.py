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
COCO_WEAPON_IDS = {43, 76}  # knife, scissors
COCO_BAG_IDS = {24, 26, 28}  # backpack, handbag, suitcase

# Fine-tuned border custom class mapping
BORDER_CLASS_MAP = {
    "person": "human",
    "human": "human",
    "pedestrian": "human",
    "man": "human",
    "woman": "human",
    "people": "human",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "motorbike": "vehicle",
    "bicycle": "vehicle",
    "bike": "vehicle",
    "vehicle": "vehicle",
    "van": "vehicle",
    "suv": "vehicle",
    "jeep": "vehicle",
    "pickup": "vehicle",
    "auto": "vehicle",
    "rickshaw": "vehicle",
    "train": "vehicle",
    "boat": "vehicle",
    "animal_drawn_cart": "vehicle",
    "animal": "animal",
    "dog": "animal",
    "cow": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cat": "animal",
    "bird": "animal",
    "weapon": "weapon",
    "knife": "weapon",
    "gun": "weapon",
    "rifle": "weapon",
    "pistol": "weapon",
    "firearm": "weapon",
    "melee": "weapon",
    "sword": "weapon",
    "large_backpack": "large_backpack",
    "backpack": "large_backpack",
    "suitcase": "large_backpack",
    "handbag": "large_backpack",
    "bag": "large_backpack",
    "luggage": "large_backpack",
    "duffel": "large_backpack",
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
    trajectory: List[List[float]] = field(default_factory=list)
    movement_flags: List[str] = field(default_factory=list)

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
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
    ):
        """
        Initialize the YOLOv8 detector with balanced confidence for CCTV surveillance.

        Args:
            model_path: Path to YOLOv8 weights (defaults to models/yolov8_custom.pt if present, else models/yolov8s.pt)
            confidence_threshold: Minimum confidence score for detections (default: 0.25 for CCTV recall)
            device: 'cuda', 'cpu', or None for auto-detection
        """
        if model_path is None:
            if os.path.exists("models/yolov8_custom.pt"):
                model_path = "models/yolov8_custom.pt"
            elif os.path.exists("models/yolov8s.pt"):
                model_path = "models/yolov8s.pt"
            else:
                model_path = "yolov8s.pt"

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
            for k, v in BORDER_CLASS_MAP.items():
                if k in cname:
                    return v

        if class_id in COCO_HUMAN_IDS:
            return "human"
        elif class_id in COCO_VEHICLE_IDS:
            return "vehicle"
        elif class_id in COCO_ANIMAL_IDS:
            return "animal"
        elif class_id in COCO_WEAPON_IDS:
            return "weapon"
        elif class_id in COCO_BAG_IDS:
            return "large_backpack"
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
                    imgsz=640,
                    iou=0.45,
                    verbose=False,
                )

                raw_entities: List[DetectedEntity] = []
                detected_threat_props: List[Dict[str, Any]] = []

                for r in pred_results:
                    boxes = r.boxes
                    if boxes is None:
                        continue

                    for i in range(len(boxes)):
                        box = boxes[i]
                        cls_id = int(box.cls[0].item())
                        cls_name = r.names.get(cls_id, "") if hasattr(r, "names") else ""
                        entity_type = self.map_class_id(cls_id, cls_name)

                        if not entity_type:
                            continue

                        score = float(box.conf[0].item())

                        # Coordinates
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1 = max(0, min(w - 1, int(xyxy[0])))
                        y1 = max(0, min(h - 1, int(xyxy[1])))
                        x2 = max(x1 + 1, min(w, int(xyxy[2])))
                        y2 = max(y1 + 1, min(h, int(xyxy[3])))
                        box_w = x2 - x1
                        box_h = y2 - y1
                        bbox = [x1, y1, x2, y2]

                        # Specific class-level confidence and dimensional sanity filtering
                        if entity_type == "human":
                            if score < 0.35:
                                continue
                        elif entity_type == "vehicle":
                            # Avoid small cardboard boxes/trash cans being detected as vehicles in CCTV
                            if score < 0.48 or (box_w * box_h) < 2800:
                                continue
                        elif entity_type == "animal":
                            # In elevated CCTV angles, pedestrians foreshortened from above
                            # often get misidentified by standard COCO as animals (dogs, sheep, cats).
                            # If the subject is upright and person-sized, reclassify as human.
                            if box_h >= 55 and (box_h / max(1, box_w)) >= 0.75:
                                entity_type = "human"
                            elif score < 0.65:
                                continue
                        elif entity_type in ("weapon", "large_backpack"):
                            if score < 0.35:
                                continue

                        # Collect weapons & large bags as threat props
                        if entity_type in ("weapon", "large_backpack"):
                            detected_threat_props.append({
                                "prop_type": entity_type,
                                "bbox": bbox,
                                "conf": score,
                                "raw_name": cls_name
                            })
                            continue

                        if entity_type not in ("human", "vehicle", "animal"):
                            continue

                        t_id = f"TMP-{uuid.uuid4().hex[:8]}"
                        foot_pt = compute_foot_point(bbox)
                        crop = frame[y1:y2, x1:x2].copy()

                        raw_entities.append(
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

                # Class-Agnostic Non-Maximum Suppression (NMS) Deduplication
                # Prevents multiple overlapping boxes stacked on the same entity
                def _compute_box_iou(b1, b2):
                    xA = max(b1[0], b2[0])
                    yA = max(b1[1], b2[1])
                    xB = min(b1[2], b2[2])
                    yB = min(b1[3], b2[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    if inter <= 0:
                        return 0.0
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    return inter / float(a1 + a2 - inter)

                raw_entities.sort(key=lambda e: e.confidence, reverse=True)
                deduped_entities: List[DetectedEntity] = []

                for ent in raw_entities:
                    overlap_found = False
                    for accepted in deduped_entities:
                        if _compute_box_iou(ent.bbox, accepted.bbox) >= 0.40:
                            overlap_found = True
                            break
                    if not overlap_found:
                        deduped_entities.append(ent)

                # Associate threat props with overlapping/nearby persons
                for h_ent in deduped_entities:
                    if h_ent.entity_type == "human":
                        hx1, hy1, hx2, hy2 = h_ent.bbox
                        pad_x = (hx2 - hx1) * 0.35
                        pad_y = (hy2 - hy1) * 0.25
                        for p in detected_threat_props:
                            px_c = (p["bbox"][0] + p["bbox"][2]) / 2.0
                            py_c = (p["bbox"][1] + p["bbox"][3]) / 2.0
                            if (hx1 - pad_x <= px_c <= hx2 + pad_x) and (hy1 - pad_y <= py_c <= hy2 + pad_y):
                                if p["prop_type"] not in h_ent.extra_props:
                                    h_ent.extra_props.append(p["prop_type"])

                return deduped_entities
            except Exception as e:
                logger.error("Error during YOLOv8 detection inference: %s. Falling back.", e)

        return results

    def detect_and_track(
        self,
        frame: np.ndarray,
        conf_override: Optional[float] = None,
        persist: bool = True,
        camera_id: Optional[str] = None,
    ) -> List[DetectedEntity]:
        """
        Convenience wrapper detecting objects and associating tracks via ByteTracker.
        Maintains camera-scoped trackers so separate camera streams track entities independently.
        """
        from .tracker import ByteTracker
        if not hasattr(self, "_camera_trackers"):
            self._camera_trackers = {}

        cam_key = camera_id or "default"
        if cam_key not in self._camera_trackers:
            self._camera_trackers[cam_key] = ByteTracker()

        raw_detections = self.detect(frame, conf_override=conf_override)
        return self._camera_trackers[cam_key].update(raw_detections, frame=frame, camera_id=cam_key)
