"""
Environmental Weather & Visibility Adaptive Analyzer for TRINETRA.

Analyzes CCTV camera frames to compute:
1. Mean Luminance (Daylight vs. Twilight vs. Pitch Dark / Night)
2. Laplacian Variance & Optical Sharpness (tau_vis)
3. Atmospheric Scattering / Contrast Ratio (Fog, Haze, Smog, Sandstorm)
4. Dynamic Streak / High-Frequency Noise (Monsoon Rain, Torrential Downpour)

Automatically classifies environmental condition into:
- CLEAR_DAY: High visibility, balanced daylight illumination
- NIGHT_LOW_LIGHT: Low luminance (< 85.0), night/dim surveillance
- FOG_HAZE: Low contrast, dense atmospheric scattering, degraded sharpness
- RAIN_STORMY: Dynamic streak noise and turbulence
- OVERHEAD_HAZY: Elevated perspective with distance haze

Recommends the optimal specialized model from the 5-model fleet:
- CLEAR_DAY -> models/yolov8_custom.pt (Master Elevated Surveillance)
- NIGHT_LOW_LIGHT -> models/yolov8_night.pt (LLVIP Night Vision & Thermal)
- FOG_HAZE -> models/yolov8_visdrone.pt + CLAHE (High-CCTV / Elevated Penetration)
- RAIN_STORMY -> models/yolov8_patrol.pt (Vehicle & Transport Focus)
- THREAT_CORROBORATION -> models/yolov8_weapon.pt (Tactical Weapons & Melee)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("weather_analyzer")


@dataclass
class EnvironmentalState:
    weather_condition: str      # CLEAR_DAY, NIGHT_LOW_LIGHT, FOG_HAZE, RAIN_STORMY, OVERHEAD_HAZY
    visibility_score: float     # 0.0 (zero visibility) to 1.0 (crystal clear)
    mean_luminance: float       # 0 to 255
    laplacian_variance: float   # Sharpness measure
    contrast_ratio: float       # Standard deviation of luminance
    is_low_light: bool          # True when luminance < 85.0
    recommended_model: str      # Target model key in 5-model fleet
    recommended_preprocessing: str  # standard, zero_dce, clahe, defog


class WeatherVisibilityAnalyzer:
    """Real-time optical analyzer assessing atmospheric conditions for adaptive model routing."""

    def __init__(self):
        self.lum_night_threshold: float = 85.0
        self.fog_contrast_threshold: float = 38.0
        self.fog_laplacian_threshold: float = 120.0
        self.clear_vis_threshold: float = 0.65

    def analyze(self, frame: np.ndarray) -> EnvironmentalState:
        """
        Analyze a single CCTV frame and return full environmental telemetry.
        """
        if frame is None or frame.size == 0:
            return EnvironmentalState(
                weather_condition="CLEAR_DAY",
                visibility_score=1.0,
                mean_luminance=128.0,
                laplacian_variance=250.0,
                contrast_ratio=50.0,
                is_low_light=False,
                recommended_model="custom",
                recommended_preprocessing="standard",
            )

        # 1. Luminance & Contrast in Grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))

        # 2. Laplacian Variance (Sharpness & High-Frequency Detail)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Visibility metric (tau_vis between 0.40 and 0.80, normalized to 0.0 - 1.0)
        tau_vis_raw = (lap_var / 250.0) * 0.40 + 0.40
        tau_vis = max(0.40, min(0.80, tau_vis_raw))
        norm_vis_score = round(float((tau_vis - 0.40) / 0.40), 2)  # 0.0 to 1.0 scale

        # 3. High-Frequency Streak Noise for Rain Detection
        # Sobel vertical vs horizontal gradient ratio checks for vertical rain streaks
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        vertical_energy = float(np.mean(np.abs(sobel_y)))
        horizontal_energy = float(np.mean(np.abs(sobel_x)))
        rain_streak_ratio = (vertical_energy / max(0.001, horizontal_energy))

        # 4. Classification of Weather / Environmental Condition
        is_low_light = bool(mean_lum < self.lum_night_threshold)

        if is_low_light:
            condition = "NIGHT_LOW_LIGHT"
            model = "night"
            prep = "zero_dce"
        elif std_lum < self.fog_contrast_threshold and lap_var < self.fog_laplacian_threshold:
            # Low contrast + blurry detail = dense fog / smog / haze
            condition = "FOG_HAZE"
            model = "visdrone"
            prep = "clahe"
        elif rain_streak_ratio > 1.45 and lap_var > 350.0 and std_lum > 40.0:
            # Strong vertical streak orientation with high gradient energy = monsoon rain
            condition = "RAIN_STORMY"
            model = "patrol"
            prep = "clahe"
        elif mean_lum > 140.0 and std_lum < 42.0:
            # High-angle elevated daylight with distance haze
            condition = "OVERHEAD_HAZY"
            model = "visdrone"
            prep = "clahe"
        else:
            condition = "CLEAR_DAY"
            model = "custom"
            prep = "standard"

        return EnvironmentalState(
            weather_condition=condition,
            visibility_score=norm_vis_score,
            mean_luminance=round(mean_lum, 2),
            laplacian_variance=round(lap_var, 2),
            contrast_ratio=round(std_lum, 2),
            is_low_light=is_low_light,
            recommended_model=model,
            recommended_preprocessing=prep,
        )


_analyzer_instance: Optional[WeatherVisibilityAnalyzer] = None


def get_weather_analyzer() -> WeatherVisibilityAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = WeatherVisibilityAnalyzer()
    return _analyzer_instance


def compute_weather_and_visibility(frame: np.ndarray) -> EnvironmentalState:
    """Convenience functional helper for real-time weather & visibility analysis."""
    return get_weather_analyzer().analyze(frame)
