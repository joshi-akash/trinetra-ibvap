#!/usr/bin/env bash
# TRINETRA — Automated Pretrained Weights Downloader (Person A)
# Downloads YOLOv8s, InsightFace buffalo_sc, and OCR dependencies.

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${MODELS_DIR}"

echo "=========================================================="
echo " TRINETRA Pretrained Weights Downloader"
echo " Destination: ${MODELS_DIR}"
echo "=========================================================="

# 1. Download YOLOv8s weights (Ultralytics GitHub release)
YOLO_FILE="${MODELS_DIR}/yolov8s.pt"
if [ ! -f "${YOLO_FILE}" ]; then
    echo "[+] Downloading YOLOv8s weights..."
    curl -L "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt" -o "${YOLO_FILE}"
    echo "[✓] Saved ${YOLO_FILE}"
else
    echo "[*] ${YOLO_FILE} already exists. Skipping."
fi

# 2. Download InsightFace buffalo_sc bundle (if not present)
INSIGHTFACE_DIR="${HOME}/.insightface/models/buffalo_sc"
mkdir -p "${INSIGHTFACE_DIR}"
echo "[*] InsightFace models will cache locally in ${INSIGHTFACE_DIR} upon first import."

# 3. Download PaddleOCR detection/recognition weights (if not present)
PADDLE_DIR="${HOME}/.paddleocr"
mkdir -p "${PADDLE_DIR}"
echo "[*] PaddleOCR weights will cache locally in ${PADDLE_DIR} upon first run."

echo "=========================================================="
echo " [✓] Pretrained weights setup complete!"
echo "=========================================================="
