# TRINETRA (IBVAP) — Sovereign Border Video Analytics Platform
**Intelligent Border Video Analytics Platform | Sovereign Air-Gapped Edge Node**  
*Developed for Border Outposts (BOP), Forward Military Installations & High-Security Perimeters*  
**Engineered by Team CodeOpia**

---

## 🎖️ System Status: 100% PRODUCTION-READY & AIR-GAPPED
* **87 / 87 Automated Tests Passing** (100% Pass Rate across unit, integration, and soak suites).
* **5 Specialized Trained YOLOv8 Models** integrated and hot-reloadable.
* **100% Offline by Design**: Zero public cloud calls, zero external CDNs, zero third-party telemetry.
* **Zero-Framework Lightweight Operator Desk**: Pure Vanilla HTML5, CSS3, ES6+ JS for maximum resilience and performance.

---

## 📑 Complete Feature Matrix (Small to Big Features)

### 1. Macro-Level / Architectural Capabilities
1. **100% Air-Gapped Sovereign Edge Node**: Fully functional in air-gapped border bunkers with zero internet/satellite uplink requirements.
2. **Multi-Model YOLOv8 Ensemble (5 Specialized Weights)**:
   - `yolov8_custom.pt` (5.91 MB): Primary active fleet model (human, worker, vehicle, large backpack). Precision: ~0.94+ mAP@0.5.
   - `yolov8_visdrone.pt` (5.92 MB): High-altitude and steep-angle detection trained on 6,471 VisDrone images across 10 classes.
   - `yolov8_weapon.pt` (5.92 MB): Threat sentry detecting firearms (`gun`) and melee weapons (`knife`, `machete`, `club`).
   - `yolov8_patrol.pt` (5.92 MB): Perimeter transport classifier (jeep, SUV, sedan, truck, van).
   - `yolov8_night.pt` (5.91 MB): Paired visible/infrared nighttime benchmark model (LLVIP) for human detection in pitch darkness.
3. **Real-Time Bi-Directional WebSocket Streaming (`/ws/alerts`)**: Sub-10ms event push between backend threads and browser canvas.
4. **4-Camera Synchronous Tactical Matrix**: 2x2 grid displaying 4 concurrent camera streams supporting RTSP, HLS (`.m3u8`), USB webcams, and local MP4s.
5. **Cryptographic SHA-256 Merkle Audit Ledger**: Tamper-proof, immutable alert hashing and hourly Merkle tree root rollup for court-martial admissibility.
6. **1-Click Forensic Evidence Package Generator**: Automatically creates signed `.ZIP` evidence archives with 30s MP4 clips, WebP thumbnails, and `evidence_metadata.json`.
7. **Sovereign 92% FIFO Storage Watchdog**: Autonomous background daemon unlinking oldest unprotected video files when disk exceeds 92% until it reaches 80%, writing an immutable `media_tombstone_log`.
8. **Dual-Stack IPv4 / IPv6 Socket Server**: Eliminates Windows localhost resolution errors (`ERR_CONNECTION_REFUSED`) by binding to `0.0.0.0` and `::` on port 8000.

### 2. Medium-Tier Subsystems & AI Modules
9. **Interactive Polygon Geofencing Studio**: Operators click and draw custom multi-point polygonal restricted zones directly on live video feeds.
10. **Ray-Casting Point-in-Polygon Mathematical Engine**: Jordan curve theorem implementation calculating exact spatial intersections between target contact points and arbitrary polygons.
11. **Passive vs. Active Alert Bifurcation**: Separates ambient background activity (livestock, benign traffic) from critical active perimeter breaches.
12. **Multi-Signal Behavioral Condition Correlation**: Correlates posture, props, and lighting into elevated threat classifications (e.g., *Night Stealth Intrusion*).
13. **Pedestrian Attribute Recognition (PAR)**:
    - Upper & lower clothing dominant color identification using HSV histogram clustering.
    - Confidence-gated neutral gender estimation (never hallucinates when ambiguous).
14. **Metric Height Estimation via Calibrated Homography**: Ground-plane perspective transformation computing real-world height in centimeters from camera calibration points.
15. **Facial Recognition System (FRS)**: RetinaFace detector + ArcFace 512-D embedding extraction with cosine similarity search against local watchlist (`data/known_suspects/`).
16. **Automatic Number Plate Recognition (ANPR)**: YOLOv8 license plate detection + PaddleOCR / EasyOCR recognition adapted for Indian number plates.
17. **Optical Anti-Tamper Sentry**:
    - Laplacian variance collapse detection for lens blur or spray blinding.
    - Histogram distribution analysis for sudden lens obstruction or blackout.
    - Reference point drift checking to detect physical camera displacement or re-angling.
18. **Environmental Adverse Weather Analyzer**: Fog, rain, and haze visibility estimation based on atmospheric attenuation models.
19. **Zero-DCE Low-Light Deep Curve Enhancement**: In-flight pixel enhancement for severely underexposed night frames without requiring paired training data.
20. **Offline Vernacular (Hindi/Hinglish) Forensic Search**: Natural language query resolution (e.g., *"safed gadi"*, *"laal kapde"*, *"andhera crawling"*, *"bandook"*).
21. **Hot-Reloadable Model Weights**: Dynamic swapping and reloading of active detector weights via `/api/training/apply-weights` without restarting the server.
22. **Continual AI Training Studio**: Operator GUI to configure epochs, batch size, learning rates, and base models for fine-tuning on local CCTV footage.

### 3. Micro-Level Features & UI Polish
23. **Automated Tactical Green NVG Filter**: Dynamic client-side SVG/CSS filter (`filter: sepia(100%) hue-rotate(90deg) saturate(350%) contrast(1.4)`) simulating Gen-3 night-vision goggles when ambient lux drops below 55/255.
24. **Dynamic Canvas Lux Estimation**: Real-time mean RGB luminous flux calculation ($0.299R + 0.587G + 0.114B$) sampled directly from the HTML5 canvas.
25. **Tactical Intrusion Modal & Synthesized Audio Siren**: Visual alert takeover modal with red pulsing perimeter and audible warning siren on unauthorized breaches.
26. **Operator Anti-Fatigue Session Cooldown**: Strict client-side suppression ensuring intrusive modal popups and audio sirens trigger *only once per operational session*, while subsequent events log silently to the ledger.
27. **45-Second Backend Alert Throttling Window**: Deduplication logic preventing identical entity detections from creating redundant database records.
28. **Bounding Box Aspect Ratio Posture Classification**:
    - $\text{AR} < 0.75$: Upright / Standing / Running.
    - $0.75 \le \text{AR} \le 1.25$: Crouching / Loitering.
    - $\text{AR} > 1.25$: Prone / Crawling Infiltration.
29. **Entity Direction Vector Tracking**: Calculates frame-to-frame displacement vectors to display heading telemetry (e.g., *Moving North-West*, *Moving Left*).
30. **Live Search Filter Chips**: Visual tags showing active search criteria (camera, time range, entity type, vernacular keywords).
31. **Video Playback Scrubbing Toolbar in Geofence Studio**: Precision controls including Play/Pause, Step Rewind (-2s), Step Forward (+2s), and Restart to freeze exact boundary frames for tracing.
32. **Vertex Undo & Reset Controls**: 1-click `Undo Point` and `Clear Polygon` buttons with live vertex count badges.
33. **Camera Fleet Activation Toggles**: Individual toggles enabling operators to show or hide specific cameras from the active 2x2 grid.
34. **Emergency Log Purge with Confirmation Modal**: Secure administrative action to wipe test entity logs while preserving audit integrity.
35. **Detection Sensitivity Selector**: Multi-tier confidence floor selector (0.20 Maximum Recall, 0.25 Balanced, 0.35 High Precision, 0.50 Strict Daylight).
36. **Zero-CDN Native UI Assets**: Embedded SVGs and vanilla CSS eliminating all external font/icon CDN network calls.
37. **National Tricolor Accent Strip & Live Military Clock**: High-visibility saffron-white-green header banner with live millisecond UTC/IST operational clock.
38. **Curated Surveillance Dataset Selector UI**: One-click UI list for selecting pre-calibrated academic datasets (VisDrone, LLVIP, Roboflow Weapons, CCPD2019).
39. **Role-Based Access Control (RBAC) Matrix Display**: Visual permission table detailing privileges across Commander, Operator, and Analyst roles.
40. **Color-Coded Bounding Box HUD**: Visual distinction between green (humans), cyan (vehicles), and high-alert red (weapons/threats).

---

## 🛠️ Tech Stack & Directory Architecture

```
trinetra/
├── run_local.py                # Standalone dual-stack production runner
├── start_production.bat        # 1-click boot script for Windows BOP server
├── requirements.txt            # Pinned dependencies
├── trinetra.db                 # Local SQLite production database
├── models/                     # 5 custom trained YOLOv8 weights (.pt)
│   ├── yolov8_custom.pt
│   ├── yolov8_visdrone.pt
│   ├── yolov8_weapon.pt
│   ├── yolov8_patrol.pt
│   └── yolov8_night.pt
├── backend/app/
│   ├── main.py                 # FastAPI application root & CORS
│   ├── api/                    # REST & WebSocket endpoints (cameras, alerts, ANPR, search, training)
│   ├── rule_engine/            # Ray-casting geofence check, bifurcation, condition wiring
│   ├── crypto_ledger/          # SHA-256 Merkle tree ledger
│   ├── storage/                # 92% FIFO disk cleaner & tombstone logger
│   ├── export_service/         # Signed ZIP evidence package creator
│   ├── search_service/         # Hindi/Hinglish query translator & synonym dictionary
│   ├── auth/                   # Local JWT auth & RBAC
│   └── static/                 # Monolithic Vanilla HTML5/CSS3/JS operator desk
├── ai_detection/               # YOLOv8 inference, PAR (color/gender), metric height, FRS, ANPR
├── ai_behavior/                # MediaPipe posture, Zero-DCE night enhancement, optical tamper sentry
└── docs/                       # Architecture specifications & integration contracts
```

---

## ⚙️ Quick Start & Production Runbook

### Windows Server / Outpost PC:
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Boot server (opens http://localhost:8000/dashboard)
python run_local.py
```

### Linux / Ubuntu Edge Server:
```bash
python3 run_local.py
# Access dashboard at http://<outpost-ip>:8000/dashboard
```

---

## 🧪 Verification & Test Coverage
```powershell
python -m pytest ai_detection/tests/ backend/tests/ -v
# Result: 87 passed, 100% success rate
```

---
*TRINETRA (IBVAP) — Sovereign Border Surveillance & Perimeter Defense System.*
