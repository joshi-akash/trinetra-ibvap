"""
Orchestration Script to Retrain All Downloaded Models for TRINETRA.

Retrains:
1. Patrol Cam Vehicles (Jeep, SUV, Sedan, Semi-truck, Van) -> models/yolov8_patrol.pt
2. Tactical Weapons (Guns, Knives, Melee) -> models/yolov8_weapon.pt
3. Overhead & Elevated CCTV Pedestrians -> models/yolov8_custom.pt

Hot-reloads the newly trained weights into the active TRINETRA live camera pipeline.
"""

import logging
import os
import shutil
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_all_downloaded")


def train_single_model(data_yaml: str, name: str, epochs: int, imgsz: int, batch: int, out_weights_path: str):
    logger.info(f"==================================================")
    logger.info(f"STARTING TRAINING: {name}")
    logger.info(f"Config: {data_yaml} | Epochs: {epochs} | ImgSz: {imgsz} | Batch: {batch}")
    logger.info(f"==================================================")

    from ultralytics import YOLO

    model = YOLO("models/yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device="cpu",
        project="runs/train",
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        val=True,
        verbose=True,
    )

    # Copy best.pt to target weights destination
    best_pt = os.path.join("runs", "train", name, "weights", "best.pt")
    if not os.path.exists(best_pt):
        # Fallback to user home runs
        user_home_runs = os.path.expanduser(os.path.join("~", "runs", "detect", "runs", "train", name, "weights", "best.pt"))
        if os.path.exists(user_home_runs):
            best_pt = user_home_runs

    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(out_weights_path), exist_ok=True)
        shutil.copy(best_pt, out_weights_path)
        logger.info(f"Successfully saved {name} weights to: {out_weights_path} ({round(os.path.getsize(out_weights_path)/(1024*1024), 2)} MB)")
        return True
    else:
        logger.warning(f"best.pt not found for {name}. Checked {best_pt}")
        return False


def main():
    start_time = time.time()
    logger.info("TRINETRA MULTI-MODEL RETRAINING PIPELINE INITIATED")

    # 1. Train Patrol Cam Vehicle Model
    patrol_yaml = "data/patrol_cam/data.yaml"
    if os.path.exists(patrol_yaml):
        train_single_model(
            data_yaml=patrol_yaml,
            name="patrol_detector",
            epochs=10,
            imgsz=320,
            batch=8,
            out_weights_path="models/yolov8_patrol.pt",
        )

    # 2. Train Elevated CCTV Pedestrian Model
    cctv_yaml = "data/cctv_elevated.yaml"
    if os.path.exists(cctv_yaml):
        train_single_model(
            data_yaml=cctv_yaml,
            name="elevated_cctv_detector",
            epochs=15,
            imgsz=416,
            batch=4,
            out_weights_path="models/yolov8_custom.pt",
        )

    # 3. Train Tactical Weapons Model (Guns & Melee)
    weapons_yaml = "data/weapons/data.yaml"
    if os.path.exists(weapons_yaml):
        train_single_model(
            data_yaml=weapons_yaml,
            name="weapons_detector",
            epochs=4,
            imgsz=320,
            batch=16,
            out_weights_path="models/yolov8_weapon.pt",
        )

    # 4. Hot-reload the live detector
    try:
        from ai_detection.training import get_auto_trainer
        trainer = get_auto_trainer()
        trainer.apply_weights("models/yolov8_custom.pt")
        logger.info("Live TRINETRA detector hot-reloaded successfully with custom weights!")
    except Exception as e:
        logger.warning(f"Could not hot-reload live detector automatically: {e}")

    elapsed = round(time.time() - start_time, 1)
    logger.info(f"==================================================")
    logger.info(f"ALL DOWNLOADED MODELS RETRAINED IN {elapsed}s!")
    logger.info(f"Models produced:")
    logger.info(f"  1. models/yolov8_patrol.pt (Perimeter Vehicles)")
    logger.info(f"  2. models/yolov8_custom.pt (Elevated CCTV Pedestrians)")
    logger.info(f"  3. models/yolov8_weapon.pt (Tactical Weapons & Melee)")
    logger.info(f"==================================================")


if __name__ == "__main__":
    main()
