import cv2
import numpy as np
import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "thresholds.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

VISIBILITY_LIMITS = config.get("visibility_limits", [0.40, 0.80])

def compute_tau_visibility(frame):
    """
    Computes visibility measure based on Laplacian variance.
    tau_vis = clip(Var(Delta I) / 250.0 * 0.40 + 0.40, 0.40, 0.80)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate tau_vis
    tau_vis = (laplacian_var / 250.0) * 0.40 + 0.40
    
    # Clip between limits
    min_val, max_val = VISIBILITY_LIMITS
    tau_vis = max(min(tau_vis, max_val), min_val)
    
    return float(tau_vis)
