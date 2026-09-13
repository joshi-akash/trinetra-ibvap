# TRINETRA (IBVAP) — Intelligent Border Video Analytics Platform
> **Smart India Hackathon | Ministry of Home Affairs (MHA) / Border Security Force (BSF)**  
> *Sovereign Offline Edge Video Surveillance Architecture for Remote Border Outposts (BOPs)*

---

## 1. Unified Monorepo Architecture

This repository is the single integrated platform combining the work of all 4 tracks:

```text
E:\project/
├── ai_detection/       # Person A: Detection, ByteTrack, PAR, Height Estimation, FRS, ANPR
│   ├── detection/      # YOLOv8 target detector & ByteTrack multi-object tracking
│   ├── height/         # Homography & perspective camera height calculator
│   ├── par/            # Pedestrian Attribute Recognition (clothing color, gender)
│   ├── frs/            # Facial recognition & known suspects vector bench
│   └── anpr/           # License plate detector & OCR engine
│
├── ai_behavior/        # Person B: Low-Visibility Vision, Pose, Tampering, Orchestrator
│   ├── night_enhancement/ # Zero-DCE low-light enhancement & dynamic visibility floor
│   ├── pose_behavior/  # MediaPipe spine angle posture classifier (crouching/prone)
│   ├── tamper/         # Optical occlusion, spray/cover, blur, reference drift
│   ├── orchestrator/   # Joint pipeline chaining detection + behavior -> FrameAnalysis
│   └── thresholds.yaml # Central operational thresholds (YAML)
│
├── backend/            # Person C: Tactical Backend, Storage, Rule Engine & Cryptography
│   ├── app/
│   │   ├── api/        # REST endpoints (Auth, Cameras, Alerts, Geofence, Entities, Search)
│   │   ├── rule_engine/# Shapely Point-in-Polygon geofence & Passive/Active bifurcation
│   │   ├── ingestion/  # RTSP intake & rolling 60s circular ring buffer
│   │   ├── alert_manager/# 30s clip slicer, WebP thumbnail & MQTT siren push
│   │   ├── search_service/# Hindi/Hinglish vernacular forensic search dictionary
│   │   ├── crypto_ledger/# Merkle tree hash chaining & tamper-proof notarization
│   │   ├── storage/    # 92% emergency FIFO cleaner & media tombstone ledger
│   │   └── static/     # Tactical browser dashboard
│   └── tests/          # Automated backend & integration pytest test suite
│
├── frontend/           # Person D: Tactical Commander Dashboard (React / Vite)
│   ├── src/            # Live 4-camera grid, alert feed, geofence polygon editor
│   └── package.json    # Frontend dependencies and Vite build scripts
│
├── infra/              # Production Deployment & Containers
│   ├── postgres/       # PostGIS 15 init schema, spatial indexes & user seeds
│   ├── mosquitto/      # Eclipse Mosquitto offline LAN MQTT broker configuration
│   ├── nginx.conf      # Reverse proxy routing frontend, API, and WebSockets
│   └── systemd/        # Linux systemd service unit for headless BOP server
│
├── models/             # Pretrained model weights & download script
├── test_footage/       # Test day/night/crouch video clips
├── docs/               # Integration contracts & OpenAPI 3.0 specs
├── docker-compose.yml  # Complete multi-container orchestration
└── run_local.py        # One-click master local platform runner
```

---

## 2. Quick Start: How to Run on Your Local Device

You can start and test the entire platform immediately using Python:

### Step 1: Run All Tests (76 Tests Across All Tracks)
```powershell
python -m pytest
```
*Expected output: 76 passed, 100% success rate.*

### Step 2: Run the Entire Platform with Tactical Dashboard
```powershell
python run_local.py
```
This single command will:
1. Initialize the local database and seed default accounts (`commander`, `admin`, `operator`).
2. Register and calibrate all 4 border outpost cameras (`CAM-01` to `CAM-04`).
3. Launch the FastAPI edge server on `http://localhost:8000`.
4. Automatically open your browser to the **Tactical Surveillance Dashboard** (`http://localhost:8000/`).

### Step 3: Run the End-to-End Simulation
To verify the full detection $\to$ passive/active bifurcation $\to$ alert package $\to$ Hindi search $\to$ HQ export lifecycle without needing physical camera hardware:
```powershell
python scripts/simulate_ingestion.py
```

---

## 3. Production Docker Deployment (For BOP Edge Server)

To deploy the entire production stack (PostGIS + Mosquitto + Backend + Nginx Frontend):
```bash
docker compose up -d
```
All services will initialize automatically in an isolated, 100% offline local area network.
