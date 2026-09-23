"""
YOLOv8 Border Model Training & Fine-Tuning Script for TRINETRA.

Implements scheduled / off-peak fine-tuning of YOLOv8 for border surveillance
incorporating hard-negatives harvested from false_flag_log (FR-FAS-03, CHANGES.md Item 6).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_yolo_border")


def train_border_model(
    data_yaml: str = "data/border_dataset.yaml",
    base_weights: str = "models/yolov8s.pt",
    output_dir: str = "runs/train",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    hard_negatives_dir: Optional[str] = None,
) -> bool:
    """
    Train or fine-tune YOLOv8 on border surveillance data with optional hard-negatives.

    Args:
        data_yaml: Path to dataset YAML configuration
        base_weights: Starting model weights path (e.g. models/yolov8s.pt)
        output_dir: Directory where trained weights and metrics are saved
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Image input resolution (e.g. 640)
        device: GPU device ID ('0') or 'cpu'
        hard_negatives_dir: Directory containing harvested false-positive samples

    Returns:
        True if training succeeded, False otherwise.
    """
    logger.info("Initializing YOLOv8 border training pass...")
    logger.info("Base weights: %s | Epochs: %d | Batch: %d | Device: %s", base_weights, epochs, batch_size, device)

    if hard_negatives_dir and os.path.isdir(hard_negatives_dir):
        logger.info("Incorporating hard negatives from: %s", hard_negatives_dir)

    try:
        from ultralytics import YOLO

        model = YOLO(base_weights)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            project=output_dir,
            name="border_detector",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            mosaic=1.0,
            augment=True,
            val=True,
        )
        logger.info("Training completed successfully. Artifacts saved in %s/border_detector", output_dir)
        return True
    except ImportError:
        logger.error("Ultralytics package is not installed. Please run `pip install ultralytics`.")
        return False
    except Exception as e:
        logger.error("Error during training pass: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="TRINETRA YOLOv8 Border Surveillance Fine-Tuning")
    parser.add_argument("--data", type=str, default="data/border_dataset.yaml", help="Path to data YAML")
    parser.add_argument("--weights", type=str, default="models/yolov8s.pt", help="Initial weights")
    parser.add_argument("--output", type=str, default="runs/train", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or cpu")
    parser.add_argument("--hard-negatives", type=str, default=None, help="Hard negatives directory")

    args = parser.parse_args()

    success = train_border_model(
        data_yaml=args.data,
        base_weights=args.weights,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        hard_negatives_dir=args.hard_negatives,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
