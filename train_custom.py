"""
TRINETRA Tactical AI Training Utility
Automated, hardware-aware YOLOv8 trainer with auto GPU/CPU detection.

Usage:
  python train_custom.py                        # Interactive menu or default weapons training
  python train_custom.py --dataset weapons     # Train weapons detector
  python train_custom.py --dataset visdrone    # Train overhead drone/CCTV detector
  python train_custom.py --dataset llvip       # Train night/thermal detector
  python train_custom.py --epochs 30 --imgsz 640
"""

import argparse
import os
import shutil
import sys
import time

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="TRINETRA Tactical AI Training Utility")
    parser.add_argument("--dataset", type=str, default="weapons", choices=["weapons", "visdrone", "llvip", "patrol"],
                        help="Dataset to train on (weapons, visdrone, llvip, patrol)")
    parser.add_argument("--model", type=str, default="models/yolov8s.pt",
                        help="Base model to fine-tune from (e.g. models/yolov8s.pt or yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (default: 30 for GPU, 10 for CPU)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Image resolution (default: 640 for GPU, 416 for CPU)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch size (default: 16 for GPU, 8 for CPU)")
    args = parser.parse_args()

    base_dir = get_base_dir()
    os.chdir(base_dir)

    print("=" * 65)
    print("        TRINETRA TACTICAL AI MODEL TRAINING ENGINE")
    print("=" * 65)

    # 1. Check Hardware (GPU vs CPU)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            device = 0
            dev_name = torch.cuda.get_device_name(0)
            print(f"[HARDWARE] GPU Detected: {dev_name} (Using CUDA:0)")
        else:
            device = "cpu"
            print("[HARDWARE] No dedicated NVIDIA GPU found. Using CPU acceleration.")
    except Exception as e:
        device = "cpu"
        has_cuda = False
        print(f"[HARDWARE] Warning: Could not detect PyTorch device ({e}). Defaulting to CPU.")

    # 2. Optimal defaults depending on hardware
    if args.epochs is None:
        args.epochs = 30 if has_cuda else 10
    if args.imgsz is None:
        args.imgsz = 640 if has_cuda else 416
    if args.batch is None:
        args.batch = 16 if has_cuda else 8

    # 3. Locate Dataset Configuration
    def resolve_dataset_yaml(folder_name, default_rel):
        candidates = [
            os.path.join(r"E:\project\datasets", folder_name, "data.yaml"),
            os.path.join(base_dir, "datasets", folder_name, "data.yaml"),
            os.path.join(base_dir, "data", folder_name, "data.yaml"),
            os.path.join(base_dir, default_rel)
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    dataset_configs = {
        "weapons": {
            "yaml": resolve_dataset_yaml("weapons", "data/weapons/data.yaml"),
            "target": os.path.join(base_dir, "models", "yolov8_weapon.pt"),
            "desc": "Tactical Weapons (Guns, Knives, Melee)"
        },
        "visdrone": {
            "yaml": resolve_dataset_yaml("visdrone", "data/visdrone/data.yaml"),
            "target": os.path.join(base_dir, "models", "yolov8_visdrone.pt"),
            "desc": "Overhead Drone & High-Mounted CCTV"
        },
        "llvip": {
            "yaml": resolve_dataset_yaml("llvip", "data/llvip/data.yaml"),
            "target": os.path.join(base_dir, "models", "yolov8_night.pt"),
            "desc": "Night Vision / Low-Light Infrared Humans"
        },
        "patrol": {
            "yaml": resolve_dataset_yaml("patrol_cam", "data/patrol_cam/data.yaml"),
            "target": os.path.join(base_dir, "models", "yolov8_patrol.pt"),
            "desc": "Perimeter Patrol Vehicles"
        }
    }

    target_cfg = dataset_configs.get(args.dataset)
    if not target_cfg or not os.path.exists(target_cfg["yaml"]):
        print(f"[ERROR] Dataset configuration not found: {target_cfg['yaml'] if target_cfg else args.dataset}")
        sys.exit(1)

    print(f"[TARGET]  {target_cfg['desc']}")
    print(f"[CONFIG]  {target_cfg['yaml']}")
    print(f"[OUTPUT]  {target_cfg['target']}")
    print(f"[PARAMS]  Epochs: {args.epochs} | ImgSize: {args.imgsz} | Batch: {args.batch} | Device: {device}")
    print("=" * 65)

    # 4. Initialize YOLO and Run Training
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Resolve base model
    base_model_path = args.model
    if not os.path.exists(base_model_path):
        if os.path.exists("models/yolov8s.pt"):
            base_model_path = "models/yolov8s.pt"
        elif os.path.exists("yolov8s.pt"):
            base_model_path = "yolov8s.pt"
        else:
            base_model_path = "yolov8s.pt"  # Ultralytics will auto-download if missing

    print(f"[MODEL] Loading base architecture: {base_model_path} ...")
    model = YOLO(base_model_path)

    run_name = f"trinetra_{args.dataset}_{int(time.time())}"
    print(f"[TRAIN] Beginning fine-tuning pass ({run_name}) ...")

    results = model.train(
        data=target_cfg["yaml"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=device,
        project="runs/trinetra",
        name=run_name,
        patience=8,
        save=True,
        exist_ok=True,
        verbose=True
    )

    # 5. Export and Save Trained Weights
    best_weights_path = os.path.join(base_dir, "runs", "trinetra", run_name, "weights", "best.pt")
    if os.path.exists(best_weights_path):
        os.makedirs(os.path.dirname(target_cfg["target"]), exist_ok=True)
        shutil.copy(best_weights_path, target_cfg["target"])
        mb_size = round(os.path.getsize(target_cfg["target"]) / (1024 * 1024), 2)
        print("\n" + "=" * 65)
        print(f"[SUCCESS] Model training complete!")
        print(f"[WEIGHTS] Best weights deployed to: {target_cfg['target']} ({mb_size} MB)")
        print(f"[HOT-RELOAD] The TRINETRA surveillance engine will load these weights automatically.")
        print("=" * 65)
    else:
        print(f"[WARNING] Training completed, but weights file not found at: {best_weights_path}")

if __name__ == "__main__":
    main()
