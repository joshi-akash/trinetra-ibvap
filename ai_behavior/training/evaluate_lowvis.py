import os
import cv2
import sys

# Ensure modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ai_behavior.night_enhancement.adaptive_threshold import compute_tau_visibility
from ai_behavior.night_enhancement.zero_dce import ZeroDCE

def evaluate_lowvis_clips(test_clips_dir):
    """
    Evaluates Zero-DCE and adaptive thresholding on staged test clips.
    """
    zero_dce = ZeroDCE()
    
    # Check if directory exists
    if not os.path.isdir(test_clips_dir):
        print(f"Test clips directory {test_clips_dir} not found.")
        return
        
    for filename in os.listdir(test_clips_dir):
        if not filename.endswith((".mp4", ".avi", ".mkv")):
            continue
            
        filepath = os.path.join(test_clips_dir, filename)
        cap = cv2.VideoCapture(filepath)
        
        print(f"Evaluating {filename}...")
        
        frame_count = 0
        enhancement_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            tau_vis = compute_tau_visibility(frame)
            
            if tau_vis < 0.65:
                enhancement_count += 1
                enhanced_frame, is_enhanced = zero_dce.enhance(frame, tau_vis)
                
                # Check for "visual_confidence_lost" (FR-LVH-05 logic could be tested here)
                if tau_vis <= 0.40:
                    print(f"  Frame {frame_count}: Visual confidence lost! tau_vis={tau_vis:.2f}")
                    
        cap.release()
        print(f"  Enhanced {enhancement_count}/{frame_count} frames.")

if __name__ == "__main__":
    # Example usage:
    # evaluate_lowvis_clips("../../test_footage")
    print("Run evaluation on test clips...")
