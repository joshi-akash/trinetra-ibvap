import math
import yaml
import os
import cv2

# Load thresholds
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "thresholds.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

SPINE_THRESHOLD = config.get("spine_inclination_threshold", 30.0)
CROUCH_SCORE = config.get("crouch_threat_score", 0.90)

class PostureClassifier:
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
            self.pose = self.mp_pose.Pose(
                static_image_mode=True, 
                min_detection_confidence=0.5
            )
            self.has_model = True
        except (ImportError, AttributeError):
            print("Warning: mediapipe.solutions not found. Running in mock mode.")
            self.mp_pose = None
            self.pose = None
            self.has_model = False

    def classify_posture(self, crop_img):
        """
        Takes a BGR image crop of a person, extracts pose, and computes spine inclination.
        Returns (posture_string, behavior_score).
        """
        if not self.has_model:
            return "normal", 0.0

        results = self.pose.process(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return "normal", 0.0
        
        landmarks = results.pose_landmarks.landmark
        
        # Get shoulders and hips
        left_sh = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_sh = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Midpoints
        x_sh = (left_sh.x + right_sh.x) / 2.0
        y_sh = (left_sh.y + right_sh.y) / 2.0
        x_hip = (left_hip.x + right_hip.x) / 2.0
        y_hip = (left_hip.y + right_hip.y) / 2.0
        
        dy = abs(y_sh - y_hip)
        dx = abs(x_sh - x_hip)
        
        if dy == 0:
            dy = 1e-6
            
        theta_rad = math.atan(dx / dy)
        theta_deg = theta_rad * (180.0 / math.pi)
        
        if theta_deg > SPINE_THRESHOLD:
            return "crouching", CROUCH_SCORE
            
        # TODO: Implement sprinting detection (velocity)
        return "normal", 0.0

    def close(self):
        if self.pose:
            self.pose.close()
