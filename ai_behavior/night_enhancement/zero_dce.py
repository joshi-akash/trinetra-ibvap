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
        Applies Zero-DCE deep curve enhancement if tau_vis < 0.65 or mean luminance < 55.
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
        elif mean_lum < 55.0:
            needs_enhancement = True

        if not needs_enhancement:
            return frame, False

        # --- 1. Zero-DCE Quadratic Curve Estimation ---
        img_norm = frame.astype(np.float32) / 255.0
        gray_norm = gray.astype(np.float32) / 255.0

        # Adaptive parameter map A(x) inversely proportional to local brightness
        alpha_map = np.clip(1.0 - gray_norm, 0.0, 0.88)
        if len(frame.shape) == 3:
            alpha_map = alpha_map[:, :, np.newaxis]

        enhanced = img_norm
        for _ in range(self.curve_iterations):
            # Iterative non-linear curve: I_{n+1} = I_n + A * I_n * (1 - I_n)
            enhanced = enhanced + alpha_map * enhanced * (1.0 - enhanced)

        enhanced_uint8 = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

        # --- 2. Adaptive CLAHE in LAB Color Space ---
        if len(frame.shape) == 3:
            lab = cv2.cvtColor(enhanced_uint8, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            final = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            # --- 3. Bilateral Edge-Preserving Denoising ---
            final = cv2.bilateralFilter(final, d=5, sigmaColor=25, sigmaSpace=25)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            final = clahe.apply(enhanced_uint8)

        return final, True
