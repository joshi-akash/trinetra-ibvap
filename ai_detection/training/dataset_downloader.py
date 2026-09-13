"""
Automated Dataset Downloader & Preprocessor for TRINETRA.

Supports pulling and formatting border surveillance datasets from Kaggle,
Roboflow, or direct HTTP/ZIP archives, generates dataset YAML descriptors,
and records dataset licensing information for compliance (PRD §10).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zipfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dataset_downloader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Default border surveillance classes
DEFAULT_BORDER_CLASSES = [
    "human",
    "vehicle",
    "animal",
    "weapon",
    "animal_drawn_cart",
    "large_backpack",
]


def generate_dataset_yaml(
    output_yaml_path: str,
    dataset_root_dir: str,
    class_names: Optional[List[str]] = None,
) -> bool:
    """
    Generate a YOLOv8 dataset configuration YAML file.

    Args:
        output_yaml_path: Destination path for the .yaml file
        dataset_root_dir: Root directory of the dataset containing images/ and labels/
        class_names: List of class labels (defaults to DEFAULT_BORDER_CLASSES)

    Returns:
        True if generated successfully, False otherwise.
    """
    classes = class_names or DEFAULT_BORDER_CLASSES
    abs_root = os.path.abspath(dataset_root_dir).replace("\\", "/")

    content = f"""# TRINETRA Border Surveillance Dataset Configuration
path: {abs_root}
train: images/train
val: images/val
test: images/test

names:
"""
    for idx, cname in enumerate(classes):
        content += f"  {idx}: {cname}\n"

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_yaml_path)), exist_ok=True)
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Generated dataset YAML at: %s", output_yaml_path)
        return True
    except Exception as e:
        logger.error("Failed to generate dataset YAML: %s", e)
        return False


def record_dataset_license(
    dataset_dir: str,
    dataset_name: str,
    license_type: str,
    source_url: str,
    notes: Optional[str] = None,
) -> bool:
    """
    Record dataset license details to a compliance manifest (PRD §10).
    """
    manifest_path = os.path.join(dataset_dir, "DATASET_LICENSE.json")
    record = {
        "dataset_name": dataset_name,
        "license_type": license_type,
        "source_url": source_url,
        "downloaded_at": str(os.path.getmtime(dataset_dir) if os.path.exists(dataset_dir) else ""),
        "notes": notes or "Compliant for TRINETRA on-premise border analytics model fine-tuning.",
    }
    try:
        os.makedirs(dataset_dir, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        logger.info("Saved dataset license manifest to %s", manifest_path)
        return True
    except Exception as e:
        logger.error("Failed to save license manifest: %s", e)
        return False


def download_from_kaggle(
    dataset_id: str,
    output_dir: str,
    unzip: bool = True,
) -> bool:
    """
    Download dataset from Kaggle using kaggle API.
    """
    logger.info("Downloading Kaggle dataset: %s to %s", dataset_id, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    try:
        import kaggle
        kaggle.api.dataset_download_files(dataset_id, path=output_dir, unzip=unzip)
        logger.info("Kaggle download finished successfully.")
        return True
    except ImportError:
        logger.error("Kaggle library not installed. Install with `pip install kaggle`.")
        return False
    except Exception as e:
        logger.error("Kaggle download failed: %s", e)
        return False


def download_from_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: str,
    model_format: str = "yolov8",
) -> bool:
    """
    Download dataset from Roboflow via Python API.
    """
    logger.info("Downloading Roboflow dataset: %s/%s v%d", workspace, project, version)
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download(model_format, location=output_dir)
        logger.info("Roboflow download finished at: %s", dataset.location)
        return True
    except ImportError:
        logger.error("Roboflow library not installed. Install with `pip install roboflow`.")
        return False
    except Exception as e:
        logger.error("Roboflow download failed: %s", e)
        return False


def extract_local_zip(
    zip_path: str,
    output_dir: str,
) -> bool:
    """
    Extract a local ZIP archive containing dataset files.
    """
    if not os.path.exists(zip_path):
        logger.error("ZIP archive does not exist: %s", zip_path)
        return False

    os.makedirs(output_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        logger.info("Extracted %s to %s", zip_path, output_dir)
        return True
    except Exception as e:
        logger.error("Failed to extract ZIP archive: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="TRINETRA Dataset Downloader & Setup Tool")
    parser.add_argument("--source", type=str, choices=["kaggle", "roboflow", "zip", "yaml_only"], default="yaml_only", help="Dataset source")
    parser.add_argument("--dataset-id", type=str, default=None, help="Kaggle dataset ID (e.g. 'owner/dataset')")
    parser.add_argument("--rf-workspace", type=str, default=None, help="Roboflow workspace")
    parser.add_argument("--rf-project", type=str, default=None, help="Roboflow project")
    parser.add_argument("--rf-version", type=int, default=1, help="Roboflow version")
    parser.add_argument("--rf-api-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--zip-file", type=str, default=None, help="Path to local ZIP archive")
    parser.add_argument("--output-dir", type=str, default="data/border_dataset", help="Output directory")
    parser.add_argument("--yaml-output", type=str, default="data/border_dataset.yaml", help="Path for generated YAML")
    parser.add_argument("--license", type=str, default="CC-BY-4.0", help="Dataset license type")

    args = parser.parse_args()

    success = True
    if args.source == "kaggle" and args.dataset_id:
        success = download_from_kaggle(args.dataset_id, args.output_dir)
    elif args.source == "roboflow" and args.rf_api_key and args.rf_workspace and args.rf_project:
        success = download_from_roboflow(
            args.rf_api_key, args.rf_workspace, args.rf_project, args.rf_version, args.output_dir
        )
    elif args.source == "zip" and args.zip_file:
        success = extract_local_zip(args.zip_file, args.output_dir)

    if success:
        generate_dataset_yaml(args.yaml_output, args.output_dir)
        record_dataset_license(
            dataset_dir=args.output_dir,
            dataset_name=args.dataset_id or args.rf_project or "border_surveillance_data",
            license_type=args.license,
            source_url=args.source,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
