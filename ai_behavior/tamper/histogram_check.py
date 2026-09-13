import cv2
import numpy as np
import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "thresholds.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

TAMPER_BLINDFOLD_RATIO = config.get("tamper_blindfold_ratio", 0.90)
TAMPER_STATIC_FREEZE_DELTA = config.get("tamper_static_freeze_delta", 0.10)

class HistogramChecker:
    def __init__(self):
        self.prev_hist = None

    def check_histogram(self, frame):
        """
        Computes the grayscale histogram.
        Flags blindfolding if > 90% of pixels fall into the top 5 intensity bins.
        Returns (histogram_flag, frame_delta).
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Sort bins descending
        sorted_hist = np.sort(hist.flatten())[::-1]
        top_5_sum = np.sum(sorted_hist[:5])
        
        histogram_flag = bool((top_5_sum / total_pixels) > TAMPER_BLINDFOLD_RATIO)
        
        frame_delta = 1.0 # default high delta
        
        if self.prev_hist is not None:
            # Compare histograms using correlation or absolute delta
            # A low delta implies high similarity (static freeze)
            correlation = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
            frame_delta = 1.0 - max(0.0, correlation) # Normalize so 0 is identical
            
        self.prev_hist = hist.copy()
        
        return histogram_flag, float(frame_delta)
