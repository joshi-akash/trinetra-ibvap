import yaml
import os
import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "thresholds.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

DRIFT_THRESHOLD = config.get("tamper_reference_drift_threshold", 0.15)

class ReferenceDriftChecker:
    def __init__(self):
        self.reference_points = None
        self.reference_descriptors = None
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def calibrate(self, reference_frame):
        """
        Calibrates the drift checker with a reference frame.
        """
        if len(reference_frame.shape) == 3:
            gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = reference_frame
            
        kp, des = self.orb.detectAndCompute(gray, None)
        self.reference_points = kp
        self.reference_descriptors = des

    def check_drift(self, frame):
        """
        Checks if the current frame has drifted significantly from the calibration.
        Returns reference_point_drift (bool).
        """
        if self.reference_descriptors is None:
            # Cannot check drift without calibration
            return False
            
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None or len(des) == 0:
            return True # Extreme drift/blur
            
        matches = self.matcher.match(self.reference_descriptors, des)
        
        if len(matches) == 0:
            return True
            
        # Calculate average distance of matches
        distances = [m.distance for m in matches]
        avg_distance = sum(distances) / len(distances)
        
        # Normalize distance (ORB max distance is typically 256)
        normalized_dist = avg_distance / 256.0
        
        return bool(normalized_dist > DRIFT_THRESHOLD)
