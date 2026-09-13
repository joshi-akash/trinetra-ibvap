# TRINETRA (IBVAP) — Production Deployment & Operations Manual
**Sovereign Offline-First Tactical AI-CCTV Video Analytics Platform**

---

## 1. System Architecture Overview

TRINETRA is designed for completely air-gapped, zero-cloud sovereign deployments at border outposts (BOP), tactical bases, critical infrastructure perimeters, and industrial warehouses.

```
+-----------------------------------------------------------------------------+
|                            TRINETRA EDGE PLATFORM                           |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [ CCTV Camera Fleet ] ---> [ Ingestion & Decoder (OpenCV / FFmpeg / HLS) ]  |
|            |                                                                |
|            v                                                                |
|  [ Tactical Low-Light Preprocessor (Zero-DCE + CLAHE Night-Vision Curve) ]   |
|            |                                                                |
|            v                                                                |
|  [ AI Vision Core: YOLOv8 (Human/Vehicle/Animal) + ByteTrack Multitracker ] |
|            |                                                                |
|            +---> [ InsightFace FRS (Face Matching >=40px, Scos >= 0.68) ]   |
|            +---> [ PaddleOCR ANPR (Indian High-Security Plate Regex) ]      |
|            +---> [ Pedestrian Attribute Recognition (Color, Height, Posture)|
|            |                                                                |
|            v                                                                |
|  [ Behavior & Spatial Engine: Geo-Fencing + Speed/Trajectory + Rules ]      |
|            |                                                                |
|            v                                                                |
|  [ Tamper-Evident SHA-256 Ledger + Evidence Store (Clips / Thumbnails) ]     |
|            |                                                                |
|            v                                                                |
|  [ FastAPI / WebSockets ] <---> [ Tactical Command Dashboard (Web/PWA) ]    |
|                                                                             |
|  [ Continual Training Hub (yt-dlp + Pseudo-Labeler + Ultralytics YOLOv8) ]  |
+-----------------------------------------------------------------------------+
```

---

## 2. Hardware Sizing & Recommended Specifications

| Deployment Tier | Hardware Profile | Target FPS | Supported Cameras |
| :--- | :--- | :--- | :--- |
| **GPU Outpost (Recommended)** | Intel Core i7 / AMD Ryzen 7, 32GB RAM, NVIDIA RTX 3060/4060 (8GB+ VRAM) or Jetson AGX Orin | 25 - 30 FPS | 8 - 16 Live Streams |
| **CPU Edge Outpost** | Intel Core i5/i7 (12th Gen+), 16GB RAM, NVMe SSD Storage | 10 - 15 FPS | 4 - 8 Live Streams |
| **Air-Gapped Rugged Unit** | Fanless Rugged PC (Advantech / Neousys), 16GB RAM, Ubuntu 22.04 LTS / Debian 12 | 10 FPS | 4 Live Streams |

---

## 3. Quickstart Deployment Methods

### Option A: Windows Bare-Metal Deployment (One-Click)

1. Open **PowerShell** as Administrator and navigate to the project directory:
   ```powershell
   cd E:\backend_cctv
   ```
2. Run the automated production launcher:
   ```powershell
   .\scripts\start_production.ps1
   ```
3. The launcher validates Python dependencies, checks `models/yolov8s.pt`, seeds the database, and launches the server on `http://0.0.0.0:8000`.

---

### Option B: Linux Bare-Metal Deployment (Native Systemd)

1. Clone or extract the repository onto the edge node:
   ```bash
   cd /opt/trinetra
   ```
2. Create and activate a dedicated virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```
3. Install systemd unit for automatic startup on boot:
   ```bash
   sudo cp infra/systemd/trinetra-native.service /etc/systemd/system/trinetra.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now trinetra.service
   ```
4. Check service status:
   ```bash
   sudo systemctl status trinetra.service
   ```

---

### Option C: Docker & Docker Compose Deployment

1. Build and launch all services with persistent volumes:
   ```bash
   docker compose up -d --build
   ```
2. View live service logs:
   ```bash
   docker compose logs -f trinetra-edge
   ```
3. Stop services:
   ```bash
   docker compose down
   ```

---

## 4. Continual AI Model Training Hub

Operators can fine-tune detection weights on-demand to adapt to site-specific lighting, camera angles, or foliage without external cloud services.

### Accessing the Training Hub:
1. Open the dashboard at `http://localhost:8000/dashboard`.
2. Click the violet gradient button in the top navigation bar: **`🧠 Train Model`**.
3. Choose the footage source:
   - **Upload Local CCTV Video File**: Browse or drag-and-drop any `.mp4`, `.mkv`, or `.avi` recording.
   - **YouTube / Live Stream URL**: Paste a live YouTube stream link (`https://www.youtube.com/watch?v=...`), SkylineWebcams URL, or HLS `.m3u8` link.
4. Set hyperparameters:
   - **Training Epochs**: Default `3` for rapid CPU training (~1 minute) or `5-10` on GPU.
   - **Max Sample Keyframes**: Default `40` frames sampled evenly across the stream.
   - **Base Weights**: `yolov8n.pt` (nano) or `yolov8s.pt` (small).
   - **Zero-DCE Low-Light Preprocessing**: Checked by default.
5. Click **`🚀 Start AI Fine-Tuning Pipeline`**.
6. Monitor the real-time telemetry terminal:
   - `INGESTING` -> `EXTRACTING_FRAMES` -> `AUTO_ANNOTATING` -> `TRAINING_EPOCHS` -> `COMPLETED`.
7. Once completed, click **`⚡ Hot-Reload Active Fleet With New Weights`** to immediately activate `models/yolov8_custom.pt` on all live cameras without restarting the server.

---

## 5. Camera Management & Stream Ingestion

TRINETRA supports any combination of IP cameras, public streams, and offline video loops:

| Stream Type | Configuration Example | Supported Protocols |
| :--- | :--- | :--- |
| **IP CCTV Camera** | `rtsp://admin:pass@192.168.1.50:554/h264Preview_01_main` | RTSP / RTP / ONVIF |
| **Webcam / Live Stream** | `https://www.skylinewebcams.com/.../lamai.html` | HLS (`.m3u8`) / SkylineWebcams |
| **Local Footage Loop** | `/footage/warehouse_night.mp4` | MP4, MKV, AVI, WebM |

---

## 6. Verification & Automated Health Checks

1. **System Health Check**:
   ```bash
   curl -s http://localhost:8000/api/health
   # Response: {"status":"healthy","system":"TRINETRA","sovereign_offline_ready":true}
   ```
2. **Automated Test Suite**:
   ```bash
   python -m pytest "E:\backend_cctv"
   # Expected result: 90 passed in ~18s (100% pass rate)
   ```

---

## 7. Default Tactical Operator Credentials

| Username | Password | Role | Privileges |
| :--- | :--- | :--- | :--- |
| `commander` | `commander123` | Commander | Full Admin, Model Training, Geo-Fence Authoring |
| `operator` | `operator123` | Operator | Live Feeds, Alert Ack, Export Evidence |
| `auditor` | `auditor123` | Auditor | Read-Only Cryptographic Audit Ledger |
