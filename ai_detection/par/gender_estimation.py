"""
Pedestrian Gender Estimation with Confidence Gating for TRINETRA PAR.

Enforces strict confidence gating (FR-PAR-02):
If gender classification confidence is below 85% (0.85), output is strictly
'neutral', never a guessed value.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

GENDER_CONFIDENCE_THRESHOLD = 0.85  # 85% confidence cutoff (FR-PAR-02)


def estimate_gender(
    person_crop: np.ndarray,
    raw_prediction: Optional[Tuple[str, float]] = None,
) -> Tuple[str, float]:
    """
    Estimate gender with strict confidence gating.

    Args:
        person_crop: BGR image patch of detected person
        raw_prediction: Optional pre-computed tuple of (predicted_gender, confidence_score)

    Returns:
        Tuple of (gender, confidence), where gender is one of ('male', 'female', 'neutral').
        If confidence < 0.85, gender is guaranteed to be 'neutral'.
    """
    if raw_prediction is not None:
        predicted_gender, conf = raw_prediction
        conf = float(conf)
        if conf < GENDER_CONFIDENCE_THRESHOLD:
            return ("neutral", conf)
        if predicted_gender.lower() in ("male", "female"):
            return (predicted_gender.lower(), conf)
        return ("neutral", conf)

    if person_crop is None or person_crop.size == 0:
        return ("neutral", 0.0)

    # In standard operation without a loaded external attribute classifier,
    # default to neutral with 0.0 confidence to guarantee compliance with FR-PAR-02.
    return ("neutral", 0.0)
