import cv2
import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "thresholds.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

TAMPER_VARIANCE_COLLAPSE = config.get("tamper_variance_collapse", 12.0)

def check_laplacian_variance(frame):
    """
    Computes Laplacian variance of the frame.
    Returns (variance, is_lens_covered).
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_lens_covered = bool(variance < TAMPER_VARIANCE_COLLAPSE)
    
    return float(variance), is_lens_covered
