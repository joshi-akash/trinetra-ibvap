"""
TRINETRA AI Detection & Recognition Stage Package.

Exposes run_detection_stage(frame: np.ndarray, camera_id: str, ...) which orchestrates:
1. YOLOv8 detection + ByteTrack tracking (human, vehicle, animal)
2. Ground-contact foot coordinate computation
3. Pedestrian Attribute Recognition (clothing color HSV clustering + confidence-gated gender)
4. Perspective geometry height estimation
5. InsightFace FRS suspect matching (>=40x40 px resolution cutoff, Scos >= 0.68)
6. PaddleOCR ANPR with Indian license plate regex validation

Outputs entities[] matching Contract 1 (FrameAnalysis.entities[]).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import time
import numpy as np
import cv2

from .anpr.plate_detector import PlateDetector
from .anpr.plate_ocr import PlateOCR, validate_indian_plate

from .detection.yolov8_detector import DetectedEntity, YOLOv8Detector, compute_foot_point
from .frs.face_matcher import FaceMatcher
from .frs.known_suspects import KnownSuspectStore
from .height.perspective_height import PerspectiveHeightEstimator, estimate_adaptive_height
from .par.clothing_color import extract_clothing_colors, extract_vehicle_color, estimate_skin_tone, estimate_posture
from .par.gender_estimation import estimate_gender

logger = logging.getLogger(__name__)

# Global singleton stage instance for high-throughput frame ingestion
_DETECTOR_INSTANCE: Optional[YOLOv8Detector] = None
_FRS_MATCHER_INSTANCE: Optional[FaceMatcher] = None
_PLATE_DETECTOR_INSTANCE: Optional[PlateDetector] = None
_PLATE_OCR_INSTANCE: Optional[PlateOCR] = None
_SUSPECT_STORE_INSTANCE: Optional[KnownSuspectStore] = None
_TRACK_PLATE_MEMORY: Dict[Tuple[str, str], str] = {}
_CAMERA_RECENT_PLATE: Dict[str, Tuple[str, float]] = {}


def get_stage_components(
    yolo_model_path: str = "models/yolov8s.pt",
    suspects_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve or initialize singleton instances of all detection models."""
    global _DETECTOR_INSTANCE, _FRS_MATCHER_INSTANCE, _PLATE_DETECTOR_INSTANCE, _PLATE_OCR_INSTANCE, _SUSPECT_STORE_INSTANCE

    if _DETECTOR_INSTANCE is None:
        _DETECTOR_INSTANCE = YOLOv8Detector(model_path=yolo_model_path)

    if _SUSPECT_STORE_INSTANCE is None:
        _SUSPECT_STORE_INSTANCE = KnownSuspectStore(storage_dir=suspects_dir) if suspects_dir else KnownSuspectStore()

    if _FRS_MATCHER_INSTANCE is None:
        _FRS_MATCHER_INSTANCE = FaceMatcher(suspect_store=_SUSPECT_STORE_INSTANCE)

    if _PLATE_DETECTOR_INSTANCE is None:
        _PLATE_DETECTOR_INSTANCE = PlateDetector()

    if _PLATE_OCR_INSTANCE is None:
        _PLATE_OCR_INSTANCE = PlateOCR()

    return {
        "detector": _DETECTOR_INSTANCE,
        "suspect_store": _SUSPECT_STORE_INSTANCE,
        "frs": _FRS_MATCHER_INSTANCE,
        "plate_detector": _PLATE_DETECTOR_INSTANCE,
        "plate_ocr": _PLATE_OCR_INSTANCE,
    }


def reload_detector(model_path: str = "models/yolov8_custom.pt") -> YOLOv8Detector:
    """Hot-reload YOLOv8 detector singleton with fine-tuned or custom weights."""
    global _DETECTOR_INSTANCE
    logger.info("Hot-reloading YOLOv8 detector fleet with model: %s", model_path)
    if _DETECTOR_INSTANCE is not None:
        _DETECTOR_INSTANCE.reload_fleet()
    else:
        _DETECTOR_INSTANCE = YOLOv8Detector(model_path=model_path)
    return _DETECTOR_INSTANCE


def run_detection_stage(
    frame: np.ndarray,
    camera_id: str,
    calibration_data: Optional[Dict[str, Any]] = None,
    is_low_light: bool = False,
    detector: Optional[YOLOv8Detector] = None,
    frs_matcher: Optional[FaceMatcher] = None,
    plate_ocr: Optional[PlateOCR] = None,
    plate_detector: Optional[PlateDetector] = None,
) -> List[Dict[str, Any]]:
    """
    Run detection, biometrics, and attribute recognition on a single camera frame.

    Args:
        frame: BGR video frame (np.ndarray)
        camera_id: Unique camera identifier (e.g. 'CAM-01')
        calibration_data: Optional calibration parameters for perspective height estimation
        is_low_light: Whether low-light / night fallback mode is active (FR-PAR-03)
        detector: Optional injected detector instance (useful for testing)
        frs_matcher: Optional injected face matcher instance
        plate_ocr: Optional injected plate OCR instance
        plate_detector: Optional injected plate detector instance

    Returns:
        List of entity dictionaries conforming strictly to Contract 1 (FrameAnalysis.entities[]).
    """
    if frame is None or frame.size == 0:
        return []

    # Use injected instances or global singletons
    comps = get_stage_components()
    det = detector or comps["detector"]
    frs = frs_matcher or comps["frs"]
    ocr = plate_ocr or comps["plate_ocr"]
    p_det = plate_detector or comps["plate_detector"]

    # Initialize height estimator if calibration data provided
    height_estimator = PerspectiveHeightEstimator(calibration_data) if calibration_data else None

    # Step 0: Tactical Low-Light & Dynamic Contrast Enhancement
    # If the frame has low luminance (< 85) or is_low_light is flagged, apply adaptive Gamma + Zero-DCE.
    # If the frame has low contrast or medium-dim lighting, apply adaptive CLAHE in LAB space.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if (len(frame.shape) == 3 and frame.shape[2] == 3) else frame
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))

    frame_is_dark = is_low_light or mean_lum < 85.0
    if frame_is_dark:
        try:
            from ai_behavior.night_enhancement.zero_dce import ZeroDCE
            _zero_dce = ZeroDCE()
            enhanced_frame, was_enhanced = _zero_dce.enhance(frame)
            processing_frame = enhanced_frame if was_enhanced else frame
            if was_enhanced:
                is_low_light = True
        except Exception:
            processing_frame = frame
    elif std_lum < 42.0 or mean_lum < 95.0:
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            processing_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception:
            processing_frame = frame
    else:
        processing_frame = frame

    # Step 1: Detect and track entities with camera scoping (backward compatible with mocks)
    try:
        detected_entities: List[DetectedEntity] = det.detect_and_track(
            processing_frame,
            camera_id=camera_id,
            is_low_light=frame_is_dark,
        )
    except TypeError:
        try:
            detected_entities = det.detect_and_track(processing_frame, camera_id=camera_id)
        except TypeError:
            detected_entities = det.detect_and_track(processing_frame)
    output_entities: List[Dict[str, Any]] = []

    for entity in detected_entities:
        e_type = entity.entity_type
        bbox = entity.bbox
        crop = entity.crop

        attributes: Dict[str, Any] = {
            "upper_color": None,
            "lower_color": None,
            "height_cm": None,
            "gender": None,
            "plate_text": None,
            "face_match": None,
            "face_name": None,
            "vehicle_type": None,
            "direction": getattr(entity, "direction", None) or "Stationary",
            "speed_kmh": getattr(entity, "speed_kmh", None) or 0.0,
            "skin_tone": None,
            "posture": None,
            "props": list(getattr(entity, "extra_props", [])),
            "trajectory": list(getattr(entity, "trajectory", [])),
            "movement_flags": list(getattr(entity, "movement_flags", [])),
            "is_low_light": is_low_light or (mean_lum < 85.0),
        }

        if e_type == "human":
            # 1. Clothing colors (PAR)
            u_col, l_col = extract_clothing_colors(crop, is_low_light=is_low_light)
            attributes["upper_color"] = u_col
            attributes["lower_color"] = l_col

            # 2. Gender estimation with confidence gating (<0.85 -> 'neutral')
            gender, _ = estimate_gender(crop)
            attributes["gender"] = gender

            # 3. Posture, Props & Skin Tone
            attributes["posture"] = estimate_posture(bbox)
            attributes["props"] = list(getattr(entity, "extra_props", []))
            attributes["skin_tone"] = estimate_skin_tone(crop, is_low_light=is_low_light)

            # 4. Height estimation via ground-plane calibration or adaptive perspective depth
            if height_estimator is not None and height_estimator.is_calibrated:
                attributes["height_cm"] = height_estimator.estimate_height(bbox)
            else:
                attributes["height_cm"] = estimate_adaptive_height(
                    bbox=bbox,
                    frame_shape=frame.shape[:2] if (frame is not None and hasattr(frame, "shape")) else None,
                    track_id=getattr(entity, "track_id", None),
                    camera_id=camera_id,
                )

            # 5. Facial Recognition & Labeling
            face_match = frs.process_person_crop(crop)
            attributes["face_match"] = face_match
            if face_match:
                attributes["face_name"] = face_match.get("name", face_match.get("suspect_id"))
            else:
                attributes["face_name"] = "Unidentified"

            if getattr(entity, "direction", None):
                attributes["direction"] = entity.direction
            elif not attributes.get("direction"):
                attributes["direction"] = "North (Advancing)"

            if getattr(entity, "speed_kmh", None) is not None:
                attributes["speed_kmh"] = entity.speed_kmh
            attributes["plate_text"] = None

        elif e_type == "vehicle":
            # Extract vehicle sub-type (car, truck, bus, motorcycle)
            raw_cls = (entity.raw_class_name or "car").lower().strip()
            if raw_cls in ("car", "truck", "bus", "motorcycle", "bicycle"):
                attributes["vehicle_type"] = raw_cls
            else:
                attributes["vehicle_type"] = "car"

            # Vehicle color (stored in vehicle_color, while upper/lower remain None per Contract 1 PAR)
            v_col = extract_vehicle_color(crop, is_low_light=is_low_light)
            attributes["vehicle_color"] = v_col

            # Direction & Speed
            if getattr(entity, "direction", None):
                attributes["direction"] = entity.direction
            elif not attributes.get("direction"):
                attributes["direction"] = "East (Moving Right)"

            if getattr(entity, "speed_kmh", None) is not None:
                attributes["speed_kmh"] = entity.speed_kmh

            # Extract plate crop, perform OCR, and retain plate across track lifetime
            t_id = getattr(entity, "track_id", None)
            plate_cache_key = (str(camera_id), str(t_id)) if t_id is not None else None

            detected_plate = None

            # 1. Check candidate plate crops from vehicle bounding box
            if hasattr(p_det, "detect_plate_candidate_crops"):
                plate_crops = p_det.detect_plate_candidate_crops(crop)
            else:
                p_c = p_det.detect_plate_crop(crop)
                plate_crops = [p_c] if p_c is not None else []

            best_plate_crop = None
            for p_crop in plate_crops:
                if p_crop is None or p_crop.size == 0:
                    continue
                if best_plate_crop is None:
                    best_plate_crop = p_crop
                ocr_res = ocr.read_plate(p_crop)
                if ocr_res is not None and ocr_res[0]:
                    if ocr_res[2] or validate_indian_plate(ocr_res[0]):
                        detected_plate = ocr_res[0]
                        best_plate_crop = p_crop
                        break

            if best_plate_crop is not None:
                attributes["plate_crop"] = best_plate_crop

            if detected_plate:
                attributes["plate_text"] = detected_plate
                if plate_cache_key:
                    _TRACK_PLATE_MEMORY[plate_cache_key] = detected_plate
                _CAMERA_RECENT_PLATE[str(camera_id)] = (detected_plate, time.time())
            elif plate_cache_key and plate_cache_key in _TRACK_PLATE_MEMORY:
                # Retain previously recognized plate for this vehicle trajectory
                attributes["plate_text"] = _TRACK_PLATE_MEMORY[plate_cache_key]
            else:
                attributes["plate_text"] = None

            attributes["is_low_light"] = bool(is_low_light)

            attributes["upper_color"] = None
            attributes["lower_color"] = None
            attributes["height_cm"] = None
            attributes["gender"] = None
            attributes["face_match"] = None
            attributes["face_name"] = None
            attributes["skin_tone"] = None
            attributes["posture"] = None

        elif e_type == "animal":
            for k in attributes:
                attributes[k] = None

        # Build Contract 1 Entity dictionary
        entity_dict: Dict[str, Any] = {
            "track_id": entity.track_id,
            "entity_type": e_type,
            "bbox": bbox,
            "confidence": round(entity.confidence, 4),
            "attributes": attributes,
        }

        output_entities.append(entity_dict)

    # Step 2: Global Frame-Level & OSD ANPR Banner Scan
    # If any primary car in the frame does not yet have a valid plate, inspect the frame banner/header
    unresolved_cars = [
        ent for ent in output_entities
        if ent["entity_type"] == "vehicle"
        and ent["attributes"].get("vehicle_type") in ("car", "truck", "bus")
        and not ent["attributes"].get("plate_text")
    ]
    if unresolved_cars and frame is not None and frame.size > 0:
        h_f, w_f = frame.shape[:2]
        # Inspect top OSD banner where commercial ANPR cameras display the recognized plate readout
        banner_region = frame[0:int(h_f * 0.35), 0:int(w_f * 0.70)]
        banner_plate = None
        if hasattr(ocr, "read_candidate_plates"):
            b_cands = ocr.read_candidate_plates(banner_region)
            if b_cands and b_cands[0][2]:
                banner_plate = b_cands[0][0]
        else:
            b_res = ocr.read_plate(banner_region)
            if b_res and (b_res[2] or validate_indian_plate(b_res[0])):
                banner_plate = b_res[0]

        if banner_plate:
            # Associate banner plate with the dominant moving car in the scene
            def _box_area(b):
                return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            unresolved_cars.sort(key=lambda e: _box_area(e.get("bbox", [0, 0, 0, 0])), reverse=True)
            primary_veh = unresolved_cars[0]
            primary_veh["attributes"]["plate_text"] = banner_plate
            v_tid = primary_veh.get("track_id")
            if v_tid is not None:
                _TRACK_PLATE_MEMORY[(str(camera_id), str(v_tid))] = banner_plate
            _CAMERA_RECENT_PLATE[str(camera_id)] = (banner_plate, time.time())

    # Step 3: Temporal smoothing from recent camera sightings (within 4.0 seconds)
    now_t = time.time()
    for ent in output_entities:
        if ent["entity_type"] == "vehicle" and not ent["attributes"].get("plate_text"):
            v_tid = ent.get("track_id")
            p_key = (str(camera_id), str(v_tid)) if v_tid is not None else None
            if p_key and p_key in _TRACK_PLATE_MEMORY:
                ent["attributes"]["plate_text"] = _TRACK_PLATE_MEMORY[p_key]
            elif ent["attributes"].get("vehicle_type") in ("car", "truck", "bus") and str(camera_id) in _CAMERA_RECENT_PLATE:
                rec_p, rec_t = _CAMERA_RECENT_PLATE[str(camera_id)]
                if (now_t - rec_t) < 4.0:
                    ent["attributes"]["plate_text"] = rec_p
                    if p_key:
                        _TRACK_PLATE_MEMORY[p_key] = rec_p


    return output_entities



__all__ = [
    "run_detection_stage",
    "get_stage_components",
    "YOLOv8Detector",
    "DetectedEntity",
    "compute_foot_point",
    "FaceMatcher",
    "KnownSuspectStore",
    "PlateDetector",
    "PlateOCR",
    "extract_clothing_colors",
    "estimate_gender",
    "PerspectiveHeightEstimator",
    "estimate_adaptive_height",
]
