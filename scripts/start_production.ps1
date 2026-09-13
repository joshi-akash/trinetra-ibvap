# ==============================================================================
# TRINETRA (IBVAP) — Production Edge Server Launch Script (PowerShell / Windows)
# Sovereign Offline-First Video Analytics Outpost
# ==============================================================================

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  TRINETRA (IBVAP) — Sovereign AI-CCTV Production Deployment Launcher" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Environment pre-flight validation
Write-Host "[1/5] Running pre-flight environment checks..." -ForegroundColor Yellow

# Check Python version
try {
    $pythonVersion = python --version 2>&1
    Write-Host "      ✓ Detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "      ✗ Error: Python 3.10+ is required but not found in PATH." -ForegroundColor Red
    exit 1
}

# 2. Check AI Model Weights
Write-Host "[2/5] Verifying sovereign offline AI model weights..." -ForegroundColor Yellow
$modelWeights = "models\yolov8s.pt"
if (Test-Path $modelWeights) {
    $size = (Get-Item $modelWeights).Length / 1MB
    Write-Host "      ✓ Primary Model weights verified: $modelWeights ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
} else {
    Write-Host "      ⚠ Primary weights ($modelWeights) not found! Checking yolov8n.pt..." -ForegroundColor Yellow
    if (Test-Path "models\yolov8n.pt") {
        Write-Host "      ✓ Fallback model weights found: models\yolov8n.pt" -ForegroundColor Green
    } else {
        Write-Host "      ✗ Error: No model weights found in models\ directory." -ForegroundColor Red
        exit 1
    }
}

# 3. Create required runtime directories
Write-Host "[3/5] Setting up secure evidence and dataset storage paths..." -ForegroundColor Yellow
$directories = @(
    "backend\storage\clips",
    "backend\storage\thumbnails",
    "backend\storage\exports",
    "backend\storage\ring_buffer",
    "data\custom_dataset\images\train",
    "data\custom_dataset\images\val",
    "data\custom_dataset\labels\train",
    "data\custom_dataset\labels\val",
    "uploads\training_media",
    "runs\train"
)
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "      ✓ Storage hierarchy initialized." -ForegroundColor Green

# 4. Initialize Database
Write-Host "[4/5] Initializing database and cryptographic tamper ledger..." -ForegroundColor Yellow
python -c "from backend.app.database import init_db; init_db(); print('      ✓ Schema initialized.')"

# 5. Launch Production Server
Write-Host "[5/5] Launching TRINETRA Edge Server on http://0.0.0.0:8000..." -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  TRINETRA Active Telemetry: http://localhost:8000/dashboard" -ForegroundColor Green
Write-Host "  Interactive OpenAPI Docs:  http://localhost:8000/docs" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

# Set environment variables for production
$env:ENVIRONMENT = "production"
$env:SOVEREIGN_OFFLINE = "true"
$env:PYTHONPATH = (Get-Location).Path

# Launch uvicorn
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
