"""
TensorRT FP16 Engine Exporter for TRINETRA AI Models.

Exports PyTorch/YOLOv8 models to high-throughput TensorRT FP16 engines
for optimized on-premise GPU workstation inference (NFR-01, NFR-02).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("export_tensorrt")


def export_to_tensorrt(
    weights_path: str = "models/yolov8s.pt",
    imgsz: Tuple[int, int] = (640, 640),
    half: bool = True,
    device: str = "0",
    workspace_gb: int = 4,
    dynamic: bool = False,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Export a YOLOv8 PyTorch model (.pt) to a TensorRT engine (.engine).

    Args:
        weights_path: Path to PyTorch model weights (.pt)
        imgsz: Input image dimensions (height, width)
        half: Enable FP16 half-precision optimization (recommended for RTX GPUs)
        device: CUDA device index ('0') or 'cpu'
        workspace_gb: Maximum TensorRT GPU workspace size in GB
        dynamic: Enable dynamic batching/dimensions
        output_dir: Destination directory for the .engine file

    Returns:
        Path to the generated .engine file, or None if export failed.
    """
    if not os.path.exists(weights_path):
        logger.error("Weights file does not exist: %s", weights_path)
        return None

    logger.info("Starting TensorRT export for: %s", weights_path)
    logger.info(
        "Parameters: imgsz=%s | half(FP16)=%s | device=%s | workspace=%dGB | dynamic=%s",
        imgsz,
        half,
        device,
        workspace_gb,
        dynamic,
    )

    try:
        from ultralytics import YOLO

        model = YOLO(weights_path)

        # Ultralytics built-in export to engine (invokes TensorRT build pipeline)
        engine_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            device=device,
            workspace=workspace_gb,
            dynamic=dynamic,
            verbose=True,
        )

        logger.info("TensorRT export successful! Engine saved at: %s", engine_path)

        if output_dir and os.path.isdir(output_dir) and engine_path:
            dest = os.path.join(output_dir, os.path.basename(engine_path))
            if dest != engine_path:
                import shutil
                shutil.move(engine_path, dest)
                logger.info("Moved engine to: %s", dest)
                return dest

        return str(engine_path)

    except ImportError:
        logger.error("Ultralytics / TensorRT is not available. Please install ultralytics and tensorrt packages.")
        return None
    except Exception as e:
        logger.error("TensorRT export encountered an error: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="TRINETRA TensorRT FP16 Engine Conversion Tool")
    parser.add_argument("--weights", type=str, default="models/yolov8s.pt", help="Path to input .pt weights")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 640], help="Image size (e.g. 640 640)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 precision (default: True)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID (default: 0)")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace memory size in GB (default: 4)")
    parser.add_argument("--dynamic", action="store_true", default=False, help="Enable dynamic batching")
    parser.add_argument("--output", type=str, default=None, help="Optional output directory for .engine file")

    args = parser.parse_args()

    size_tuple = (args.imgsz[0], args.imgsz[1]) if len(args.imgsz) >= 2 else (args.imgsz[0], args.imgsz[0])

    engine = export_to_tensorrt(
        weights_path=args.weights,
        imgsz=size_tuple,
        half=args.fp16,
        device=args.device,
        workspace_gb=args.workspace,
        dynamic=args.dynamic,
        output_dir=args.output,
    )

    sys.exit(0 if engine else 1)


if __name__ == "__main__":
    main()
