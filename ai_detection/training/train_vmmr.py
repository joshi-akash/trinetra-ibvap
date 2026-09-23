import os
import shutil
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def format_vmmr_for_yolo():
    logger.info("Formatting VMMR Classification Dataset for YOLOv8...")
    
    # User's extracted paths
    source_dir = Path(r"E:\project\ai_detection\data\vehicles\Dataset\SubsetVMMR")
    yolo_base = Path(r"E:\project\ai_detection\data\vehicles\yolo_format")
    
    train_dir = yolo_base / "train"
    val_dir = yolo_base / "val"
    
    # If already formatted, skip
    if train_dir.exists() and val_dir.exists():
        logger.info("Dataset already appears formatted. Skipping copy step.")
        return yolo_base
        
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    classes = [d for d in source_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(classes)} vehicle classes. Processing and splitting (80/20)...")
    
    for cls_dir in classes:
        cls_name = cls_dir.name
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        
        if not images:
            continue
            
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Create class folders in train and val
        (train_dir / cls_name).mkdir(exist_ok=True)
        (val_dir / cls_name).mkdir(exist_ok=True)
        
        for img in train_imgs:
            shutil.copy(str(img), str(train_dir / cls_name / img.name))
            
        for img in val_imgs:
            shutil.copy(str(img), str(val_dir / cls_name / img.name))
            
    logger.info("Successfully formatted dataset for classification!")
    return yolo_base

def train_vmmr_model():
    data_path = format_vmmr_for_yolo()
    
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        return
        
    logger.info("Loading base YOLOv8n-cls (Classification) model...")
    # NOTE: Using the -cls model for classification, not the object detector!
    model = YOLO("yolov8n-cls.pt")
    
    logger.info("Starting intensive classification training for Vehicle Makes and Models...")
    model.train(
        data=str(data_path),
        epochs=10,
        imgsz=224,  # Standard for classification
        batch=4,    # Low batch size to prevent OOM
        workers=0,  # No multiprocessing to prevent Windows crashes
        name="vmmr_classifier_run",
        project=r"E:\backend_cctv\runs\train",
        exist_ok=True
    )
    
    best_weights = Path(r"E:\backend_cctv\runs\train\vmmr_classifier_run\weights\best.pt")
    target_path = Path(r"E:\backend_cctv\models\vmmr_yolov8n_cls.pt")
    
    if best_weights.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(best_weights), str(target_path))
        logger.info(f"Successfully saved VMMR Classifier to: {target_path}")

if __name__ == "__main__":
    train_vmmr_model()
