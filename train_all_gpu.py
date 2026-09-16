"""
TRINETRA Tactical Surveillance - GPU Batch Training Orchestrator
Sequentially trains all specialized models on the local NVIDIA GeForce RTX GPU.

Models trained:
1. Tactical Weapons (Guns, Knives, Melee)        -> models/yolov8_weapon.pt
2. VisDrone Overhead & Drone Surveillance         -> models/yolov8_visdrone.pt
3. LLVIP Night-Vision / Thermal Infrared Humans   -> models/yolov8_night.pt
4. Perimeter Patrol Vehicles                      -> models/yolov8_patrol.pt
5. Primary Fleet Elevated Surveillance Master     -> models/yolov8_custom.pt

Usage:
  python train_all_gpu.py
  python train_all_gpu.py --epochs 25 --batch 16 --imgsz 640
"""

import argparse
import os
import shutil
import sys
import time

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            print(f"[GPU DETECTED] {device_name} ({total_mem} GB VRAM) - CUDA Acceleration ACTIVE")
            return 0
        else:
            print("[WARNING] CUDA not detected in PyTorch. Running on CPU (slower).")
            print("To enable GPU, run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            return "cpu"
    except Exception as e:
        print(f"[WARNING] Hardware check warning: {e}. Defaulting to CPU.")
        return "cpu"

def train_model(name: str, yaml_path: str, output_path: str, base_model: str, epochs: int, imgsz: int, batch: int, device):
    print("\n" + "=" * 70)
    print(f" >>> STARTING GPU TRAINING: {name}")
    print(f" >>> Dataset : {yaml_path}")
    print(f" >>> Target  : {output_path}")
    print(f" >>> Settings: Epochs={epochs} | ImgSz={imgsz} | Batch={batch} | Device={device}")
    print("=" * 70)

    if not os.path.exists(yaml_path):
        print(f"[SKIP] Dataset config not found: {yaml_path}")
        return False

    from ultralytics import YOLO

    model = YOLO(base_model)
    run_name = f"gpu_{name.lower().replace(' ', '_')}_{int(time.time())}"

    start_t = time.time()
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/trinetra_gpu",
        name=run_name,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.15,
        patience=10,
        save=True,
        exist_ok=True,
        verbose=True
    )
    elapsed = round(time.time() - start_t, 1)

    # Resolve and copy best.pt
    best_pt = os.path.join("runs", "trinetra_gpu", run_name, "weights", "best.pt")
    if not os.path.exists(best_pt):
        user_runs = os.path.expanduser(os.path.join("~", "runs", "detect", "runs", "trinetra_gpu", run_name, "weights", "best.pt"))
        if os.path.exists(user_runs):
            best_pt = user_runs

    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(best_pt, output_path)
        mb_size = round(os.path.getsize(output_path) / (1024 * 1024), 2)
        print(f"[SUCCESS] {name} trained in {elapsed}s. Saved to: {output_path} ({mb_size} MB)")
        return True
    else:
        print(f"[WARNING] Training completed for {name}, but best.pt not found at: {best_pt}")
        return False

def main():
    parser = argparse.ArgumentParser(description="TRINETRA GPU Batch Training Orchestrator")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per model (default: 20)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16 for 6GB RTX 3050)")
    parser.add_argument("--model", type=str, default="models/yolov8s.pt", help="Base model architecture")
    args = parser.parse_args()

    base_dir = get_base_dir()
    os.chdir(base_dir)

    print("=" * 70)
    print("      TRINETRA TACTICAL AI - FULL GPU FLEET RETRAINING SUITE")
    print("=" * 70)

    device = check_gpu()

    # Model Pipeline Definition
    pipeline = [
        {
            "name": "Tactical Weapons Detector",
            "yaml": os.path.join(base_dir, "data", "weapons", "data.yaml"),
            "out": os.path.join(base_dir, "models", "yolov8_weapon.pt"),
            "base": args.model
        },
        {
            "name": "VisDrone Overhead & High-CCTV Detector",
            "yaml": os.path.join(base_dir, "data", "visdrone", "data.yaml"),
            "out": os.path.join(base_dir, "models", "yolov8_visdrone.pt"),
            "base": args.model
        },
        {
            "name": "LLVIP Night Vision & Thermal Humans",
            "yaml": os.path.join(base_dir, "data", "llvip", "data.yaml"),
            "out": os.path.join(base_dir, "models", "yolov8_night.pt"),
            "base": args.model
        },
        {
            "name": "Perimeter Patrol Vehicles",
            "yaml": os.path.join(base_dir, "data", "patrol_cam", "data.yaml"),
            "out": os.path.join(base_dir, "models", "yolov8_patrol.pt"),
            "base": args.model
        },
        {
            "name": "Elevated Surveillance Fleet Master",
            "yaml": os.path.join(base_dir, "data", "surveillance_sample.yaml"),
            "out": os.path.join(base_dir, "models", "yolov8_custom.pt"),
            "base": args.model
        }
    ]

    total_start = time.time()
    results_summary = []

    for item in pipeline:
        success = train_model(
            name=item["name"],
            yaml_path=item["yaml"],
            output_path=item["out"],
            base_model=item["base"],
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device
        )
        results_summary.append((item["name"], success))

    total_time = round((time.time() - total_start) / 60, 2)
    print("\n" + "=" * 70)
    print(f" >>> ALL GPU TRAINING PIPELINES FINISHED in {total_time} minutes")
    print("=" * 70)
    for name, success in results_summary:
        status_str = "SUCCESS (SAVED & DEPLOYED)" if success else "SKIPPED/FAILED"
        print(f" - {name:<45} : {status_str}")

    # Hot-reload active server
    try:
        from ai_detection.training.auto_trainer import get_auto_trainer
        trainer = get_auto_trainer()
        trainer.apply_weights("models/yolov8_custom.pt")
        print("[HOT-RELOAD] Live surveillance fleet updated with newly trained weights!")
    except Exception as e:
        print(f"[HOT-RELOAD] Server will pick up new weights on next cycle: {e}")

if __name__ == "__main__":
    main()
