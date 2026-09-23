import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def convert_voc_to_yolo():
    logger.info("Starting VOC to YOLO conversion for Fire & Smoke dataset...")
    
    # User's extracted paths
    base_dir = Path(r"E:\project\ai_detection\data\fire")
    ann_dir = base_dir / "Annotations" / "Annotations"
    img_dir = base_dir / "Datacluster Fire and Smoke Sample" / "Datacluster Fire and Smoke Sample"
    
    # YOLO format output directories
    yolo_base = base_dir / "yolo_format"
    images_dir = yolo_base / "images"
    labels_dir = yolo_base / "labels"
    
    if not ann_dir.exists() or not img_dir.exists():
        logger.error("Could not find the extracted XMLs or Images!")
        return None

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    xml_files = glob.glob(str(ann_dir / "*.xml"))
    if not xml_files:
        logger.error(f"No XML files found in {ann_dir}")
        return None
        
    class_map = {"fire": 0, "smoke": 1}
    
    processed = 0
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        size = root.find('size')
        if size is None:
            continue
            
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        if w == 0 or h == 0:
            continue
            
        filename = root.find('filename').text
        img_path = img_dir / filename
        
        if not img_path.exists():
            continue
            
        # Create YOLO label file
        txt_name = Path(xml_file).stem + ".txt"
        txt_path = labels_dir / txt_name
        
        has_boxes = False
        with open(txt_path, 'w') as out_file:
            for obj in root.iter('object'):
                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 1:
                    continue
                    
                obj_name = obj.find('name').text.lower()
                class_id = class_map.get(obj_name, -1)
                
                if class_id == -1:
                    continue
                    
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)
                
                # Convert to YOLO normalized
                b_center_x = (xmin + xmax) / 2.0 / w
                b_center_y = (ymin + ymax) / 2.0 / h
                b_width = (xmax - xmin) / w
                b_height = (ymax - ymin) / h
                
                out_file.write(f"{class_id} {b_center_x:.6f} {b_center_y:.6f} {b_width:.6f} {b_height:.6f}\n")
                has_boxes = True
                
        if has_boxes:
            shutil.copy(str(img_path), str(images_dir / filename))
            processed += 1
            
    logger.info(f"Successfully converted {processed} images/labels to YOLO format.")
    
    # Generate YAML
    yaml_path = yolo_base / "fire_smoke.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"path: {yolo_base.absolute().as_posix()}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("nc: 2\n")
        f.write("names: ['fire', 'smoke']\n")
        
    return yaml_path

def train_fire_model():
    yaml_path = convert_voc_to_yolo()
    if not yaml_path:
        return
        
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        return
        
    logger.info("Loading base YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    logger.info("Starting intensive Fire & Smoke Detection training...")
    model.train(
        data=str(yaml_path),
        epochs=10,
        imgsz=320,  # Lowered to prevent OOM
        batch=2,    # Lowered to prevent OOM
        workers=0,  # No multiprocessing
        name="fire_detector_run",
        project=r"E:\backend_cctv\runs\train",
        exist_ok=True
    )
    
    best_weights = Path(r"E:\backend_cctv\runs\train\fire_detector_run\weights\best.pt")
    target_path = Path(r"E:\backend_cctv\models\fire_yolov8n.pt")
    
    if best_weights.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(best_weights), str(target_path))
        logger.info(f"Successfully saved Fire Detector to: {target_path}")

if __name__ == "__main__":
    train_fire_model()
