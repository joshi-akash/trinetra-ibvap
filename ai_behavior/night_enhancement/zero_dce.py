import yaml
import os
import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "thresholds.yaml")
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {}

ZERO_DCE_THRESHOLD = config.get("zero_dce_threshold", 0.65)

class ZeroDCE:
    """
    Tactical Low-Light and Zero-Lux Enhancement Engine for TRINETRA.
    Combines Zero-DCE (Deep Curve Estimation) non-linear illumination mapping,
    Adaptive CLAHE in LAB color space, and bilateral noise suppression to enable
    high-confidence detection in near-pitch darkness.
    """
    def __init__(self, curve_iterations: int = 4, clahe_clip: float = 2.8):
        self.curve_iterations = curve_iterations
        self.clahe_clip = clahe_clip

    def enhance(self, frame: np.ndarray, tau_vis: float = None):
        """
        Applies tactical adaptive low-light enhancement (Dynamic Gamma + Zero-DCE + LAB CLAHE).
        Triggers if tau_vis < ZERO_DCE_THRESHOLD or mean luminance < 85.0.
        Returns (enhanced_frame, is_enhanced).
        """
        if frame is None or frame.size == 0:
            return frame, False

        # Compute mean luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_lum = float(np.mean(gray))

        # Check triggers: visibility measure or low luminance (night/dim conditions)
        needs_enhancement = False
        if tau_vis is not None and tau_vis < ZERO_DCE_THRESHOLD:
            needs_enhancement = True
        elif mean_lum < 85.0:
            needs_enhancement = True

        if not needs_enhancement:
            return frame, False

        # --- 1. Dynamic Adaptive Gamma Illumination Boost ---
        norm_lum = max(12.0, min(mean_lum, 140.0)) / 255.0
        # Darker scenes get stronger non-linear lift (gamma < 1.0)
        gamma = float(np.clip(np.log(0.45) / np.log(norm_lum), 0.40, 0.88))
        inv_gamma = 1.0 / gamma
        lut_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        gamma_boosted = cv2.LUT(frame, lut_table)

        # --- 2. Zero-DCE Quadratic Curve Iteration ---
        img_norm = gamma_boosted.astype(np.float32) / 255.0
        gray_boosted = cv2.cvtColor(gamma_boosted, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else gamma_boosted
        gray_norm = gray_boosted.astype(np.float32) / 255.0

        # Adaptive illumination parameter map A(x)
        alpha_map = np.clip(1.0 - gray_norm, 0.0, 0.85)
        if len(frame.shape) == 3:
            alpha_map = alpha_map[:, :, np.newaxis]

        enhanced = img_norm
        for _ in range(self.curve_iterations):
            # Iterative non-linear curve: I_{n+1} = I_n + A * I_n * (1 - I_n)
            enhanced = enhanced + alpha_map * enhanced * (1.0 - enhanced)

        enhanced_uint8 = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

        # --- 3. Adaptive CLAHE in LAB Color Space ---
        if len(frame.shape) == 3:
            lab = cv2.cvtColor(enhanced_uint8, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=max(self.clahe_clip, 2.5), tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            final = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            # --- 4. Bilateral Edge-Preserving Denoising ---
            final = cv2.bilateralFilter(final, d=5, sigmaColor=20, sigmaSpace=20)
        else:
            clahe = cv2.createCLAHE(clipLimit=max(self.clahe_clip, 2.5), tileGridSize=(8, 8))
            final = clahe.apply(enhanced_uint8)

        return final, True
