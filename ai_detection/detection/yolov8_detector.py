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


FLEET_MODEL_PATHS = {
    "custom": "models/yolov8_custom.pt",
    "weapon": "models/yolov8_weapon.pt",
    "night": "models/yolov8_night.pt",
    "patrol": "models/yolov8_patrol.pt",
    "visdrone": "models/yolov8_visdrone.pt",
    "camo": "models/camo_yolov8n.pt",
    "drone": "models/drone_yolov8n.pt",
    "fire": "models/fire_yolov8n.pt",
    "luggage": "models/luggage_yolov8n.pt",
    "plate": "models/plate_yolov8n.pt",
    "vmmr": "models/vmmr_yolov8n_cls.pt",
}


class YOLOv8Detector:
    """
    YOLOv8 Object Detector for bbox inference, foot-point computation,
    and 5-Model Fleet Ensemble routing (Custom, Weapons, Night Vision, Patrol, VisDrone).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.15,
        device: Optional[str] = None,
        enable_ensemble: bool = True,
    ):
        """
        Initialize the YOLOv8 detector with balanced confidence for CCTV surveillance.

        Args:
            model_path: Path to primary YOLOv8 weights (defaults to models/yolov8_custom.pt)
            confidence_threshold: Minimum confidence score for detections (default: 0.15)
            device: 'cuda', 'cpu', or None for auto-detection
            enable_ensemble: Whether to enable the 5-model fleet ensemble routing
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
        self.enable_ensemble = enable_ensemble
        self.model = None
        self._is_mock = False
        self.fleet_models: Dict[str, Any] = {}

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load primary Ultralytics YOLO model; fallback gracefully if not found."""
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info("Loaded primary YOLOv8 model from %s", self.model_path)
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

    def get_fleet_model(self, key: str) -> Optional[Any]:
        """Lazy-load and cache an individual fleet model weights instance."""
        if key in self.fleet_models:
            return self.fleet_models[key]

        weights_path = FLEET_MODEL_PATHS.get(key)
        if not weights_path or not os.path.exists(weights_path):
            return None

        try:
            from ultralytics import YOLO
            m = YOLO(weights_path)
            if self.device:
                m.to(self.device)
            self.fleet_models[key] = m
            logger.info("Loaded fleet ensemble model [%s] from %s", key, weights_path)
            return m
        except Exception as e:
            logger.debug("Could not load fleet ensemble model [%s]: %s", key, e)
            return None

    def reload_fleet(self, key: Optional[str] = None) -> None:
        """Hot-reload cached fleet models (e.g. after fresh training)."""
        if key:
            self.fleet_models.pop(key, None)
            if key == "custom":
                self._load_model()
        else:
            self.fleet_models.clear()
            self._load_model()
        logger.info("Fleet models reloaded (target: %s)", key or "ALL")

    def map_class_id(self, class_id: int, class_name: Optional[str] = None) -> Optional[str]:
        """
        Map raw YOLO class ID or name to TRINETRA canonical entity_type:
        'human', 'vehicle', 'animal', 'weapon', or 'large_backpack'.
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

    def _parse_boxes(
        self,
        pred_results,
        frame: np.ndarray,
        conf_floor: float = 0.25,
        forced_entity_type: Optional[str] = None,
    ) -> Tuple[List[DetectedEntity], List[Dict[str, Any]]]:
        """Convert raw prediction results into candidate entities and threat props."""
        h, w = frame.shape[:2]
        raw_entities: List[DetectedEntity] = []
        threat_props: List[Dict[str, Any]] = []

        for r in pred_results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0].item())
                cls_name = r.names.get(cls_id, "") if hasattr(r, "names") else ""
                entity_type = forced_entity_type or self.map_class_id(cls_id, cls_name)

                if not entity_type:
                    continue

                score = float(box.conf[0].item())
                if score < conf_floor:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, min(w - 1, int(xyxy[0])))
                y1 = max(0, min(h - 1, int(xyxy[1])))
                x2 = max(x1 + 1, min(w, int(xyxy[2])))
                y2 = max(y1 + 1, min(h, int(xyxy[3])))
                box_w = x2 - x1
                box_h = y2 - y1
                bbox = [x1, y1, x2, y2]

                if entity_type == "human":
                    if score < 0.30:
                        continue
                elif entity_type == "vehicle":
                    if score < 0.15:
                        continue
                elif entity_type == "animal":
                    if box_h >= 55 and (box_h / max(1, box_w)) >= 0.75:
                        entity_type = "human"
                    elif score < 0.65:
                        continue
                elif entity_type in ("weapon", "large_backpack"):
                    if score < 0.25:
                        continue

                # Collect weapons & large bags as threat props
                if entity_type in ("weapon", "large_backpack"):
                    threat_props.append({
                        "prop_type": entity_type,
                        "bbox": bbox,
                        "conf": score,
                        "raw_name": cls_name,
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

        return raw_entities, threat_props

    def detect(
        self,
        frame: np.ndarray,
        conf_override: Optional[float] = None,
        is_low_light: bool = False,
        enable_ensemble: Optional[bool] = None,
    ) -> List[DetectedEntity]:
        """
        Run 5-model fleet ensemble detection inference on a single frame.

        Args:
            frame: Input BGR image (np.ndarray)
            conf_override: Optional override for confidence threshold
            is_low_light: Whether frame is dark / night mode (routes to yolov8_night.pt)
            enable_ensemble: Toggle multi-model fleet corroboration

        Returns:
            List of DetectedEntity objects with bounding boxes and foot points.
        """
        if frame is None or frame.size == 0:
            return []

        use_ensemble = self.enable_ensemble if enable_ensemble is None else enable_ensemble
        conf = conf_override if conf_override is not None else self.confidence_threshold
        raw_entities: List[DetectedEntity] = []
        detected_threat_props: List[Dict[str, Any]] = []

        # 1. Primary Model Inference (models/yolov8_custom.pt or active backbone)
        if self.model is not None and not self._is_mock:
            try:
                pred_results = self.model.predict(
                    source=frame,
                    conf=conf,
                    imgsz=640,
                    iou=0.45,
                    verbose=False,
                )
                m_ents, m_threats = self._parse_boxes(pred_results, frame, conf_floor=conf)
                raw_entities.extend(m_ents)
                detected_threat_props.extend(m_threats)
            except Exception as e:
                logger.error("Error during primary YOLOv8 inference: %s", e)

        # 2. Multi-Model Fleet Corroboration & Weather-Adaptive Routing
        if use_ensemble and not self._is_mock:
            try:
                from ai_behavior.environmental import compute_weather_and_visibility
                env_state = compute_weather_and_visibility(frame)
            except Exception:
                env_state = None

            # Check environmental condition
            frame_is_dark = is_low_light or (env_state.is_low_light if env_state else False)
            weather_cond = env_state.weather_condition if env_state else ("NIGHT_LOW_LIGHT" if frame_is_dark else "CLEAR_DAY")

            # (A) LLVIP Night Vision & Low-Light Humans (models/yolov8_night.pt)
            # Active in pitch darkness, dusk/night, or stormy rain conditions
            if frame_is_dark or weather_cond in ("NIGHT_LOW_LIGHT", "RAIN_STORMY"):
                night_m = self.get_fleet_model("night")
                if night_m is not None:
                    try:
                        n_preds = night_m.predict(source=frame, conf=0.28, imgsz=640, verbose=False)
                        n_ents, _ = self._parse_boxes(n_preds, frame, conf_floor=0.28, forced_entity_type="human")
                        raw_entities.extend(n_ents)
                    except Exception as ex:
                        logger.debug("Night vision ensemble pass error: %s", ex)

            # (B) Tactical Weapons & Melee Threat Items (models/yolov8_weapon.pt)
            weapon_m = self.get_fleet_model("weapon")
            if weapon_m is not None:
                try:
                    w_preds = weapon_m.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
                    _, w_threats = self._parse_boxes(w_preds, frame, conf_floor=0.25)
                    detected_threat_props.extend(w_threats)
                except Exception as ex:
                    logger.debug("Weapon ensemble pass error: %s", ex)

            # (C) Perimeter Patrol Vehicles (models/yolov8_patrol.pt)
            has_vehicle_candidate = any(e.entity_type == "vehicle" for e in raw_entities)
            patrol_m = self.get_fleet_model("patrol")
            if patrol_m is not None and (has_vehicle_candidate or weather_cond == "RAIN_STORMY" or len(raw_entities) == 0):
                try:
                    p_preds = patrol_m.predict(source=frame, conf=0.36, imgsz=640, verbose=False)
                    p_ents, _ = self._parse_boxes(p_preds, frame, conf_floor=0.36, forced_entity_type="vehicle")
                    raw_entities.extend(p_ents)
                except Exception as ex:
                    logger.debug("Patrol vehicle ensemble pass error: %s", ex)

            # (D) VisDrone Overhead & Elevated CCTV (models/yolov8_visdrone.pt)
            # Highly effective in Fog/Haze/Smog scattering and high-angle elevated perspectives
            visdrone_m = self.get_fleet_model("visdrone")
            if visdrone_m is not None:
                try:
                    v_conf = 0.26 if weather_cond in ("FOG_HAZE", "OVERHEAD_HAZY") else 0.30
                    v_preds = visdrone_m.predict(source=frame, conf=v_conf, imgsz=640, verbose=False)
                    v_ents, _ = self._parse_boxes(v_preds, frame, conf_floor=v_conf)
                    raw_entities.extend(v_ents)
                except Exception as ex:
                    logger.debug("VisDrone ensemble pass error: %s", ex)

            # (E) Fire/Smoke Hazard Detection
            fire_m = self.get_fleet_model("fire")
            if fire_m is not None:
                try:
                    f_preds = fire_m.predict(source=frame, conf=0.35, imgsz=640, verbose=False)
                    for r in f_preds:
                        if r.boxes:
                            for box in r.boxes:
                                if float(box.conf[0]) >= 0.35:
                                    cls_name = r.names.get(int(box.cls[0]), "fire_hazard") if hasattr(r, 'names') else "fire_hazard"
                                    detected_threat_props.append({"prop_type": f"environmental_{cls_name}", "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist(), "conf": float(box.conf[0]), "raw_name": cls_name})
                except Exception as ex:
                    logger.debug("Fire pass error: %s", ex)

            # (F) Camouflage Suspects
            camo_m = self.get_fleet_model("camo")
            if camo_m is not None:
                try:
                    c_preds = camo_m.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
                    c_ents, _ = self._parse_boxes(c_preds, frame, conf_floor=0.25, forced_entity_type="human")
                    for e in c_ents:
                        e.extra_props.append("camouflaged")
                    raw_entities.extend(c_ents)
                except Exception as ex:
                    logger.debug("Camo pass error: %s", ex)

            # (G) Abandoned Luggage Detection
            lug_m = self.get_fleet_model("luggage")
            if lug_m is not None:
                try:
                    l_preds = lug_m.predict(source=frame, conf=0.30, imgsz=640, verbose=False)
                    _, l_threats = self._parse_boxes(l_preds, frame, conf_floor=0.30, forced_entity_type="large_backpack")
                    for t in l_threats:
                        t["prop_type"] = "unattended_luggage"
                    detected_threat_props.extend(l_threats)
                except Exception as ex:
                    logger.debug("Luggage pass error: %s", ex)

            # (H) Drone UAV Detection
            drone_m = self.get_fleet_model("drone")
            if drone_m is not None:
                try:
                    d_preds = drone_m.predict(source=frame, conf=0.35, imgsz=640, verbose=False)
                    for r in d_preds:
                        if r.boxes:
                            for box in r.boxes:
                                if float(box.conf[0]) >= 0.35:
                                    detected_threat_props.append({"prop_type": "unauthorized_uav", "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist(), "conf": float(box.conf[0]), "raw_name": "drone"})
                except Exception as ex:
                    logger.debug("Drone pass error: %s", ex)

            # (I) VMMR Classification for Vehicles
            vmmr_m = self.get_fleet_model("vmmr")
            if vmmr_m is not None:
                try:
                    for e in raw_entities:
                        if e.entity_type == "vehicle":
                            v_preds = vmmr_m.predict(source=e.crop, imgsz=224, verbose=False)
                            for r in v_preds:
                                if hasattr(r, 'probs') and r.probs is not None:
                                    top1_idx = r.probs.top1
                                    top1_conf = r.probs.top1conf.item()
                                    if top1_conf > 0.40:
                                        make_model = r.names[top1_idx]
                                        e.extra_props.append(f"Make/Model: {make_model}")
                except Exception as ex:
                    logger.debug("VMMR pass error: %s", ex)

        if not raw_entities and not detected_threat_props:
            return []

        # 3. Class-Agnostic Non-Maximum Suppression (NMS) Deduplication
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

        # 4. Associate threat props with overlapping/nearby persons
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

    def detect_and_track(
        self,
        frame: np.ndarray,
        conf_override: Optional[float] = None,
        persist: bool = True,
        camera_id: Optional[str] = None,
        is_low_light: bool = False,
        enable_ensemble: Optional[bool] = None,
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

        raw_detections = self.detect(
            frame,
            conf_override=conf_override,
            is_low_light=is_low_light,
            enable_ensemble=enable_ensemble,
        )
        return self._camera_trackers[cam_key].update(raw_detections, frame=frame, camera_id=cam_key)

