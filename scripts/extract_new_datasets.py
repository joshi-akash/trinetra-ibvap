"""
TRINETRA Dataset Unpacker & Manager
Extracts and registers all datasets from E:\project\datasets for GPU training.
"""

import os
import sys
import shutil
import zipfile
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if r"E:\backend_cctv" not in sys.path:
    sys.path.insert(0, r"E:\backend_cctv")

DATASETS_ROOT = r"E:\project\datasets"

def extract_zip(zip_path, target_dir):
    print(f"\n[EXTRACT] Extracting {os.path.basename(zip_path)} -> {target_dir} ...")
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(target_dir)
    print(f"[DONE] Extracted {os.path.basename(zip_path)} successfully.")

def fix_yaml(yaml_path, base_path):
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        data['path'] = base_path.replace('\\', '/')
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"[YAML UPDATED] {yaml_path} set path -> {data['path']}")

def main():
    print("=" * 70)
    print("      TRINETRA DATASET EXTRACTION & CONFIGURATION ENGINE")
    print(f"      Source Location: {DATASETS_ROOT}")
    print("=" * 70)

    # 1. Weapons Dataset
    weapons_zip = os.path.join(DATASETS_ROOT, "BIGGER WEAPONS.v2i.yolov8.zip")
    weapons_dir = os.path.join(DATASETS_ROOT, "weapons")
    if os.path.exists(weapons_zip) and not os.path.exists(os.path.join(weapons_dir, "train")):
        extract_zip(weapons_zip, weapons_dir)
        fix_yaml(os.path.join(weapons_dir, "data.yaml"), weapons_dir)
    else:
        print(f"[EXISTS] Weapons dataset already present at: {weapons_dir}")

    # 2. Patrol Cam Vehicles Dataset
    patrol_zip = os.path.join(DATASETS_ROOT, "Patrol Cam.v1i.yolov8.zip")
    patrol_dir = os.path.join(DATASETS_ROOT, "patrol_cam")
    if os.path.exists(patrol_zip) and not os.path.exists(os.path.join(patrol_dir, "train")):
        extract_zip(patrol_zip, patrol_dir)
        fix_yaml(os.path.join(patrol_dir, "data.yaml"), patrol_dir)
    else:
        print(f"[EXISTS] Patrol Cam dataset already present at: {patrol_dir}")

    # 3. VisDrone Overhead & CCTV Dataset (Full Extraction)
    visdrone_zip = os.path.join(DATASETS_ROOT, "VisDrone2019-DET-train.zip")
    visdrone_dir = os.path.join(DATASETS_ROOT, "visdrone")
    if os.path.exists(visdrone_zip) and not os.path.exists(os.path.join(visdrone_dir, "images")):
        print(f"\n[EXTRACT] Parsing and converting VisDrone-DET annotations from: {visdrone_zip} ...")
        from ai_detection.training.prepare_visdrone import prepare_visdrone
        prepare_visdrone(visdrone_zip, output_dir=visdrone_dir, max_samples=6471)
        fix_yaml(os.path.join(visdrone_dir, "data.yaml"), visdrone_dir)
    else:
        print(f"[EXISTS] VisDrone dataset already present at: {visdrone_dir}")

    # 4. LLVIP Night Vision & Thermal Humans
    llvip_zip = os.path.join(DATASETS_ROOT, "LLVIP.zip")
    llvip_dir = os.path.join(DATASETS_ROOT, "llvip")
    if os.path.exists(llvip_zip) and not os.path.exists(os.path.join(llvip_dir, "images")):
        print(f"\n[EXTRACT] Parsing and converting LLVIP night surveillance dataset from: {llvip_zip} ...")
        from ai_detection.training.prepare_llvip import prepare_llvip_dataset
        prepare_llvip_dataset(llvip_zip, output_dir=llvip_dir, train_count=3000, val_count=500)
        fix_yaml(os.path.join(llvip_dir, "data.yaml"), llvip_dir)
    else:
        print(f"[EXISTS] LLVIP dataset already present at: {llvip_dir}")

    # 5. LOL Dataset (Low-Light Paired)
    lol_zip = os.path.join(DATASETS_ROOT, "archive (1).zip")
    lol_dir = os.path.join(DATASETS_ROOT, "lol_dataset")
    if os.path.exists(lol_zip) and not os.path.exists(lol_dir):
        extract_zip(lol_zip, lol_dir)
    else:
        print(f"[EXISTS] LOL dataset already present at: {lol_dir}")

    print("\n" + "=" * 70)
    print("      ALL DATASETS CONFIGURED IN: " + DATASETS_ROOT)
    print("=" * 70)

if __name__ == "__main__":
    main()
