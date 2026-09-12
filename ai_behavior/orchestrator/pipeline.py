import time
import datetime
import yaml
import os

from ai_behavior.night_enhancement.adaptive_threshold import compute_tau_visibility
from ai_behavior.night_enhancement.zero_dce import ZeroDCE
from ai_behavior.pose_behavior.posture_classifier import PostureClassifier
from ai_behavior.pose_behavior.prop_detection import PropDetector
from ai_behavior.tamper.laplacian_check import check_laplacian_variance
from ai_behavior.tamper.histogram_check import HistogramChecker
from ai_behavior.tamper.reference_drift import ReferenceDriftChecker

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "thresholds.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

FREEZE_TIME_S = config.get("tamper_static_freeze_time_s", 3.0)
FREEZE_DELTA = config.get("tamper_static_freeze_delta", 0.10)

class FrameOrchestrator:
    def __init__(self):
        self.zero_dce = ZeroDCE()
        self.posture_classifier = PostureClassifier()
        self.prop_detector = PropDetector()
        self.histogram_checker = HistogramChecker()
        self.reference_drift = ReferenceDriftChecker()
        
        self.static_freeze_start = None

    def process_frame(self, frame, camera_id, frame_ref, run_detection_stage_callback):
        """
        Executes the AI behavior pipeline and assembles the FrameAnalysis object.
        run_detection_stage_callback is expected to be a function provided by Person A 
        that takes an image and returns a list of detected entity dicts.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # 1. Night Enhancement & Visibility
        tau_vis = compute_tau_visibility(frame)
        is_low_light = (tau_vis < 0.65)
        
        enhanced_frame, is_enhanced = self.zero_dce.enhance(frame, tau_vis)
        processing_frame = enhanced_frame if is_enhanced else frame
        
        # 2. Tamper Detection
        variance, is_lens_covered = check_laplacian_variance(processing_frame)
        histogram_flag, frame_delta = self.histogram_checker.check_histogram(processing_frame)
        reference_drift_flag = self.reference_drift.check_drift(processing_frame)
        
        static_freeze = False
        if frame_delta < FREEZE_DELTA:
            if self.static_freeze_start is None:
                self.static_freeze_start = time.time()
            elif (time.time() - self.static_freeze_start) >= FREEZE_TIME_S:
                static_freeze = True
        else:
            self.static_freeze_start = None
            
        is_tampered = is_lens_covered or histogram_flag or reference_drift_flag or static_freeze

        # 3. Object Detection (Person A's stage)
        raw_entities = run_detection_stage_callback(processing_frame)
        
        # 4. Pose & Prop Analytics
        enriched_entities = []
        for entity in raw_entities:
            # We only process humans for posture and props
            if entity.get("entity_type") == "human":
                # Assuming 'crop' is provided by detection stage or we crop using bbox
                crop_img = entity.get("crop", processing_frame) # Fallback to full frame if crop missing
                
                posture, behavior_score = self.posture_classifier.classify_posture(crop_img)
                props = self.prop_detector.detect_props(crop_img)
                
                # Append attributes
                if "attributes" not in entity:
                    entity["attributes"] = {}
                    
                entity["attributes"]["posture"] = posture
                entity["attributes"]["props"] = props
                
                # Remove crop from final JSON to save bandwidth
                if "crop" in entity:
                    del entity["crop"]
                    
            enriched_entities.append(entity)

        # 5. Assemble FrameAnalysis Object
        frame_analysis = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "frame_ref": frame_ref,
            "frame_quality": {
                "low_light": is_low_light,
                "enhanced": is_enhanced,
                "tau_visibility": float(tau_vis),
                "tamper_signal": {
                    "is_tampered": bool(is_tampered),
                    "laplacian_variance": float(variance),
                    "histogram_flag": bool(histogram_flag),
                    "reference_point_drift": bool(reference_drift_flag),
                    "static_freeze": bool(static_freeze)
                }
            },
            "entities": enriched_entities
        }
        
        return frame_analysis

    def close(self):
        self.posture_classifier.close()
