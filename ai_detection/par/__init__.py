"""PAR (Pedestrian Attribute Recognition) module for TRINETRA."""
from .clothing_color import extract_clothing_colors, CANONICAL_COLORS
from .gender_estimation import estimate_gender, GENDER_CONFIDENCE_THRESHOLD

__all__ = [
    "extract_clothing_colors",
    "CANONICAL_COLORS",
    "estimate_gender",
    "GENDER_CONFIDENCE_THRESHOLD",
]
