import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_drone_model():
    logger.info("Initializing Drone Detection YOLOv8 Training...")
    
    # Fix for Kaggle nested extraction
    data_yaml = Path(r"E:\project\ai_detection\data\drone\drone-detection-new.v5-new-train.yolov8\data.yaml")
    
    if not data_yaml.exists():
        logger.error(f"Could not find {data_yaml}.")
        return
        
    # Auto-patch the YAML file to use absolute paths so YOLO doesn't fail
    with open(data_yaml, 'r') as f:
        yaml_content = f.read()
        
    yaml_content = yaml_content.replace('../train/images', str(data_yaml.parent / 'train' / 'images'))
    yaml_content = yaml_content.replace('../valid/images', str(data_yaml.parent / 'valid' / 'images'))
    yaml_content = yaml_content.replace('../test/images', str(data_yaml.parent / 'test' / 'images'))
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        return
        
    logger.info("Loading base YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    logger.info("Starting intensive Drone Detection training...")
    model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=320,  # Lowered from 640 to prevent OOM
        batch=2,    # Lowered from 4 to prevent OOM
        workers=0,  # No multiprocessing to prevent Windows crashes
        name="drone_detector_run",
        project=r"E:\backend_cctv\runs\train",
        exist_ok=True
    )
    
    best_weights = Path(r"E:\backend_cctv\runs\train\drone_detector_run\weights\best.pt")
    target_path = Path(r"E:\backend_cctv\models\drone_yolov8n.pt")
    
    if best_weights.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(best_weights), str(target_path))
        logger.info(f"Successfully saved Drone Detector to: {target_path}")

if __name__ == "__main__":
    train_drone_model()
