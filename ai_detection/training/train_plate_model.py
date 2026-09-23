"""
TRINETRA Plate YOLOv8 Trainer

Downloads or generates a dataset of license plates and trains a dedicated YOLOv8n
model for highly accurate Automated Number Plate Recognition (ANPR) bounding box localization.
Output weights will be saved to `models/plate_yolov8n.pt`.
"""

import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Root definitions
ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "ai_detection" / "data" / "plates"

def train_plate_detector():
    logger.info("Initializing License Plate YOLOv8 Detector Training...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Path where the user will extract the Roboflow dataset
    base_dir = Path(r"E:\project\ai_detection\data\plates")
    
    if not base_dir.exists():
        logger.error(f"Could not find {base_dir}. Please create it and extract the ZIP inside.")
        return
        
    yaml_files = list(base_dir.rglob("data.yaml"))
    if not yaml_files:
        logger.error(f"Could not find data.yaml inside {base_dir}!")
        return
        
    data_yaml = yaml_files[0]
    logger.info(f"Found dataset config at: {data_yaml}")
    
    # Auto-patch the YAML file to use absolute paths so YOLO doesn't fail
    with open(data_yaml, 'r') as f:
        yaml_content = f.read()
        
    yaml_content = yaml_content.replace('../train/images', (data_yaml.parent / 'train' / 'images').as_posix())
    yaml_content = yaml_content.replace('../valid/images', (data_yaml.parent / 'valid' / 'images').as_posix())
    yaml_content = yaml_content.replace('../test/images', (data_yaml.parent / 'test' / 'images').as_posix())
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is not installed. Please install it to train the model.")
        return

    logger.info("Loading base YOLOv8n model...")
    model = YOLO("yolov8n.pt") 
    
    logger.info("Starting intensive ANPR Plate Detector training...")
    results = model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=320,
        batch=2,
        workers=0,
        name="plate_detector_run",
        project=str(ROOT / "runs" / "train"),
        exist_ok=True
    )
    
    logger.info("Training complete. Exporting weights...")
    best_weights = ROOT / "runs" / "train" / "plate_detector_run" / "weights" / "best.pt"
    
    target_path = MODELS_DIR / "plate_yolov8n.pt"
    if best_weights.exists():
        shutil.copy(str(best_weights), str(target_path))
        logger.info(f"Successfully saved specialized ANPR Plate Detector to: {target_path}")
    else:
        logger.error("Failed to locate best.pt after training.")
        
if __name__ == "__main__":
    train_plate_detector()
