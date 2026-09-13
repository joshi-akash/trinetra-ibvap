"""
Pedestrian Clothing Color Extractor for TRINETRA PAR.

Clusters upper-torso and lower-body pixels in HSV color space into a fixed canonical
color vocabulary: ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white', 'gray', 'brown', 'unknown'].
Enforces IR/monochrome night fallback to 'unknown' (FR-PAR-03).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CANONICAL_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "black",
    "white",
    "gray",
    "brown",
    "unknown",
]


def bgr_to_hsv_numpy(bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to HSV (H: 0-179, S: 0-255, V: 0-255) using OpenCV if available,
    or pure vectorized NumPy fallback.
    """
    try:
        import cv2
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    except ImportError:
        pass

    # NumPy vectorized BGR to HSV
    b = bgr[:, :, 0].astype(np.float32) / 255.0
    g = bgr[:, :, 1].astype(np.float32) / 255.0
    r = bgr[:, :, 2].astype(np.float32) / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 1e-5)
    mask_g = (cmax == g) & (delta > 1e-5)
    mask_b = (cmax == b) & (delta > 1e-5)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Convert to OpenCV HSV scale: H in [0, 179], S in [0, 255], V in [0, 255]
    h_cv = (h / 2.0).astype(np.uint8)
    s_cv = np.zeros_like(cmax, dtype=np.float32)
    valid_mask = cmax > 1e-5
    s_cv[valid_mask] = (delta[valid_mask] / cmax[valid_mask]) * 255.0
    s_cv = s_cv.astype(np.uint8)
    v_cv = (cmax * 255.0).astype(np.uint8)

    return np.dstack((h_cv, s_cv, v_cv))


def classify_hsv_pixel(h: int, s: int, v: int) -> str:
    """
    Classify a single pixel (H: 0-179, S: 0-255, V: 0-255) into a canonical color bin.
    """
    # Achromatic checks first: Black, White, Gray
    if v < 45:
        return "black"
    if s < 38:
        if v > 190:
            return "white"
        return "gray"

    # Chromatic classification by Hue
    # Red wraps around 0-10 and 170-179
    if (h <= 10) or (h >= 170):
        if s > 60 and v > 50:
            # Check for brown/dark red
            if v < 110 and s > 80:
                return "brown"
            return "red"
        return "gray"
    elif 11 <= h <= 25:
        if v < 120:
            return "brown"
        return "orange"
    elif 26 <= h <= 35:
        return "yellow"
    elif 36 <= h <= 85:
        return "green"
    elif 86 <= h <= 135:
        return "blue"
    elif 136 <= h <= 169:
        return "purple"

    return "gray"


def get_dominant_color(patch: np.ndarray) -> str:
    """
    Determine dominant canonical color in an image patch.
    """
    if patch is None or patch.size == 0:
        return "unknown"

    hsv = bgr_to_hsv_numpy(patch)
    h_chan = hsv[:, :, 0].flatten()
    s_chan = hsv[:, :, 1].flatten()
    v_chan = hsv[:, :, 2].flatten()

    # Subsample pixels for speed
    step = max(1, len(h_chan) // 500)
    h_sample = h_chan[::step]
    s_sample = s_chan[::step]
    v_sample = v_chan[::step]

    votes: Dict[str, int] = {c: 0 for c in CANONICAL_COLORS}

    for h_val, s_val, v_val in zip(h_sample, s_sample, v_sample):
        color_label = classify_hsv_pixel(int(h_val), int(s_val), int(v_val))
        votes[color_label] += 1

    # Remove 'unknown' from consideration if other colors exist
    del votes["unknown"]

    # Dominant color
    dominant = max(votes, key=votes.get)  # type: ignore
    if votes[dominant] == 0:
        return "unknown"

    return dominant


def extract_clothing_colors(
    person_crop: np.ndarray,
    is_low_light: bool = False,
) -> Tuple[str, str]:
    """
    Extract upper and lower clothing colors from a human bounding box crop.

    Args:
        person_crop: BGR image crop of a detected person
        is_low_light: Flag indicating low-light or IR/monochrome conditions (FR-PAR-03)

    Returns:
        Tuple of (upper_color, lower_color) drawn strictly from CANONICAL_COLORS.
    """
    # Under IR/monochrome or low-light conditions, fallback to unknown (FR-PAR-03)
    if is_low_light:
        return ("unknown", "unknown")

    if person_crop is None or person_crop.size == 0:
        return ("unknown", "unknown")

    h, w = person_crop.shape[:2]
    if h < 20 or w < 10:
        return ("unknown", "unknown")

    # Upper torso: 15% to 50% height, central 60% width
    u_y1, u_y2 = int(h * 0.15), int(h * 0.50)
    u_x1, u_x2 = int(w * 0.20), int(w * 0.80)
    upper_patch = person_crop[u_y1:u_y2, u_x1:u_x2]

    # Lower body: 50% to 90% height, central 60% width
    l_y1, l_y2 = int(h * 0.50), int(h * 0.90)
    l_x1, l_x2 = int(w * 0.20), int(w * 0.80)
    lower_patch = person_crop[l_y1:l_y2, l_x1:l_x2]

    upper_color = get_dominant_color(upper_patch)
    lower_color = get_dominant_color(lower_patch)

    return (upper_color, lower_color)
