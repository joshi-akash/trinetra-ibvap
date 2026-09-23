#!/usr/bin/env bash
# ==============================================================================
# TRINETRA (IBVAP) — Production Edge Server Launch Script (Bash / Linux)
# Sovereign Offline-First Video Analytics Outpost
# ==============================================================================

set -euo pipefail

echo -e "\033[1;36m================================================================================\033[0m"
echo -e "\033[1;36m  TRINETRA (IBVAP) — Sovereign AI-CCTV Production Deployment Launcher\033[0m"
echo -e "\033[1;36m================================================================================\033[0m"

# 1. Environment pre-flight validation
echo -e "\033[1;33m[1/5] Running pre-flight environment checks...\033[0m"
if ! command -v python3 &> /dev/null; then
    echo -e "\033[1;31m      ✗ Error: python3 is not installed or not in PATH.\033[0m"
    exit 1
fi
echo -e "\033[1;32m      ✓ Detected: $(python3 --version)\033[0m"

# 2. Check AI Model Weights
echo -e "\033[1;33m[2/5] Verifying sovereign offline AI model weights...\033[0m"
if [ -f "models/yolov8s.pt" ]; then
    echo -e "\033[1;32m      ✓ Primary Model weights verified: models/yolov8s.pt\033[0m"
elif [ -f "models/yolov8n.pt" ]; then
    echo -e "\033[1;32m      ✓ Fallback model weights verified: models/yolov8n.pt\033[0m"
else
    echo -e "\033[1;31m      ✗ Error: No model weights found in models/ directory.\033[0m"
    exit 1
fi

# 3. Create required runtime directories
echo -e "\033[1;33m[3/5] Setting up secure evidence and dataset storage paths...\033[0m"
mkdir -p backend/storage/clips \
         backend/storage/thumbnails \
         backend/storage/exports \
         backend/storage/ring_buffer \
         data/custom_dataset/images/train \
         data/custom_dataset/images/val \
         data/custom_dataset/labels/train \
         data/custom_dataset/labels/val \
         uploads/training_media \
         runs/train
echo -e "\033[1;32m      ✓ Storage hierarchy initialized.\033[0m"

# 4. Initialize Database
echo -e "\033[1;33m[4/5] Initializing database and cryptographic tamper ledger...\033[0m"
python3 -c "from backend.app.database import init_db; init_db(); print('      ✓ Schema initialized.')"

# 5. Launch Production Server
echo -e "\033[1;33m[5/5] Launching TRINETRA Edge Server on http://0.0.0.0:8000...\033[0m"
echo -e "\033[1;36m================================================================================\033[0m"
echo -e "\033[1;32m  TRINETRA Active Telemetry: http://localhost:8000/dashboard\033[0m"
echo -e "\033[1;32m  Interactive OpenAPI Docs:  http://localhost:8000/docs\033[0m"
echo -e "\033[1;36m================================================================================\033[0m"

export ENVIRONMENT="production"
export SOVEREIGN_OFFLINE="true"
export PYTHONPATH="$(pwd)"

exec python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
