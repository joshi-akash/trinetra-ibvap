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

import numpy as np

from .anpr.plate_detector import PlateDetector
from .anpr.plate_ocr import PlateOCR
from .detection.yolov8_detector import DetectedEntity, YOLOv8Detector, compute_foot_point
from .frs.face_matcher import FaceMatcher
from .frs.known_suspects import KnownSuspectStore
from .height.perspective_height import PerspectiveHeightEstimator
from .par.clothing_color import extract_clothing_colors
from .par.gender_estimation import estimate_gender

logger = logging.getLogger(__name__)

# Global singleton stage instance for high-throughput frame ingestion
_DETECTOR_INSTANCE: Optional[YOLOv8Detector] = None
_FRS_MATCHER_INSTANCE: Optional[FaceMatcher] = None
_PLATE_DETECTOR_INSTANCE: Optional[PlateDetector] = None
_PLATE_OCR_INSTANCE: Optional[PlateOCR] = None
_SUSPECT_STORE_INSTANCE: Optional[KnownSuspectStore] = None


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

    # Step 1: Detect and track entities
    detected_entities: List[DetectedEntity] = det.detect_and_track(frame)
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
        }

        if e_type == "human":
            # 1. Clothing colors (PAR)
            u_col, l_col = extract_clothing_colors(crop, is_low_light=is_low_light)
            attributes["upper_color"] = u_col
            attributes["lower_color"] = l_col

            # 2. Gender estimation with confidence gating (<0.85 -> 'neutral')
            gender, _ = estimate_gender(crop)
            attributes["gender"] = gender

            # 3. Height estimation via ground-plane calibration
            if height_estimator is not None and height_estimator.is_calibrated:
                attributes["height_cm"] = height_estimator.estimate_height(bbox)
            else:
                attributes["height_cm"] = None

            # 4. Facial Recognition (InsightFace RetinaFace + ArcFace)
            face_match = frs.process_person_crop(crop)
            attributes["face_match"] = face_match

            # plate_text remains None for humans
            attributes["plate_text"] = None

        elif e_type == "vehicle":
            # Extract plate crop and perform OCR
            plate_crop = p_det.detect_plate_crop(crop)
            if plate_crop is not None:
                ocr_res = ocr.read_plate(plate_crop)
                if ocr_res is not None:
                    attributes["plate_text"] = ocr_res[0]

            # Biometrics/human attributes remain None for vehicles
            attributes["upper_color"] = None
            attributes["lower_color"] = None
            attributes["height_cm"] = None
            attributes["gender"] = None
            attributes["face_match"] = None

        elif e_type == "animal":
            # All biometrics remain None for animals
            attributes["upper_color"] = None
            attributes["lower_color"] = None
            attributes["height_cm"] = None
            attributes["gender"] = None
            attributes["plate_text"] = None
            attributes["face_match"] = None

        # Build Contract 1 Entity dictionary
        entity_dict: Dict[str, Any] = {
            "track_id": entity.track_id,
            "entity_type": e_type,
            "bbox": bbox,
            "confidence": round(entity.confidence, 4),
            "attributes": attributes,
        }

        output_entities.append(entity_dict)

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
]
