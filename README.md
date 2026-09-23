<div align="center">

# 🛡️ TRINETRA (IBVAP)

### Intelligent Border Video Analytics Platform
**Sovereign · Air-Gapped · Edge-Deployed Perimeter Intelligence**

[![Status](https://img.shields.io/badge/status-active-brightgreen?style=for-the-badge)](#)
[![Offline First](https://img.shields.io/badge/deployment-100%25%20air--gapped-blueviolet?style=for-the-badge)](#)
[![Tests](https://img.shields.io/badge/tests-87%2F87%20passing-success?style=for-the-badge)](#-testing)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8%20fleet-FF6F00?style=for-the-badge)](#-ai-model-fleet)
[![License](https://img.shields.io/badge/license-add--your--license-lightgrey?style=for-the-badge)](#-license)

**No cloud. No CDN. No external API calls. Ever.**

*Built for security forces operating where WAN connectivity is unreliable, restricted, or intentionally absent.*

</div>

<br>

> 🎯 **Mission:** Give a single machine at the edge — no GPU required — everything it needs to detect, verify, log and prove a perimeter breach, entirely on its own.

<br>

## 📚 Table of Contents

| | | | |
|---|---|---|---|
| 🎯 [Why TRINETRA](#-why-trinetra) | ✨ [Key Features](#-key-features) | 🏗️ [Architecture](#️-architecture) | 📁 [Repository Structure](#-repository--file-structure) |
| 🧠 [AI Model Fleet](#-ai-model-fleet) | 🧮 [Core Algorithms](#-core-algorithms) | 📊 [Datasets](#-datasets) | 🚀 [Getting Started](#-getting-started) |
| 🖥️ [Usage Guide](#️-usage-guide) | 🔌 [API Reference](#-api--websocket-reference) | ✅ [Testing](#-testing) | ⚠️ [Limitations](#️-known-limitations) |
| 🗺️ [Roadmap](#️-roadmap) | 📖 [Glossary](#-glossary) | 📄 [License](#-license) | 🤝 [Contributing](#-contributing) |

<br>

---

## 🎯 Why TRINETRA

Border outposts sit at the end of thin or non-existent communication lines. Commercial video-analytics products assume a cloud back end and a stable WAN link — assumptions that simply don't hold at a BOP. TRINETRA is built as a **single sovereign edge node**: one machine, standard CPU hardware, that performs detection, rule evaluation, alerting, evidence preservation and search entirely on-site.

<table>
<tr><th align="left">🚧 Operational Problem</th><th align="left">✅ TRINETRA's Response</th></tr>
<tr><td>No WAN/cloud; dashboards that depend on CDNs hang or fail</td><td>100% air-gapped design · vanilla HTML5/CSS3/ES6+ frontend · inline SVG icons · zero third-party calls</td></tr>
<tr><td>Intrusions happen in darkness; camouflaged/crawling intruders are missed by generic models</td><td>LLVIP-trained night model · real-time luminance estimation · synthetic green NVG filter · aspect-ratio posture analysis</td></tr>
<tr><td>Operators get flooded by continuous detections at 30 FPS</td><td>45-second backend suppression window + once-per-session client modal/siren cooldown</td></tr>
<tr><td>Evidence must survive legal scrutiny; disks are finite</td><td>SHA-256 hash-chained ledger with hourly Merkle roots · signed evidence ZIPs · tombstoned deletions</td></tr>
<tr><td>Operators think/speak in Hindi/Hinglish, not query syntax</td><td>Local vernacular search with a Hindi/Hinglish synonym dictionary</td></tr>
</table>

> 💡 **Design principles:** Sovereignty first &nbsp;·&nbsp; CPU-viable edge inference &nbsp;·&nbsp; Evidence integrity by construction &nbsp;·&nbsp; Low cognitive load for the operator

<br>

---

## ✨ Key Features

<table>
<tr>
<td width="33%" valign="top">

### 📹 Detection & Monitoring
- **4-Camera Synchronous Matrix** — responsive 2×2 grid, live streams or local file upload
- **Interactive Polygon Geofencing** — draw arbitrary (including non-convex) perimeter zones
- **Camera Tamper Detection** — Laplacian-variance & histogram sentries flag defocus/lens interference

</td>
<td width="33%" valign="top">

### 🚨 Alerting & Response
- **Real-Time Threat Modal & Siren** — WebSocket-driven high-priority alerts
- **Operator Anti-Fatigue Cooldown** — siren/modal once per session, ledger stays complete
- **Immutable Audit Ledger** — live feed of camera, class, confidence, rule, timestamp

</td>
<td width="33%" valign="top">

### 🔍 Search & Evidence
- **Vernacular Search** — English, Hindi & Hinglish, resolved fully on-device
- **Forensic ZIP Export** — one click: video + thumbnail + SHA-256 metadata
- **Autonomous Storage Management** — FIFO watchdog keeps disk between 80–92%

</td>
</tr>
</table>

> ✅ **Verified:** 87 automated tests (100% passing) — API endpoints, geofence math, bifurcation rules, low-light handling, soak loads.

<br>

---

## 🏗️ Architecture

Two execution domains coexist in one process:

| Domain | Owns | Notes |
|---|---|---|
| 🧵 **Synchronous worker threads** | Frame ingestion, YOLO inference | CPU-heavy; must never be blocked by network I/O |
| ⚡ **FastAPI/Uvicorn async event loop** | REST, static delivery, WebSocket channel | Bridged via `asyncio.run_coroutine_threadsafe()` |

```
┌──────────────────────────────────────────────────────────────────────────┐
│  📹 VIDEO SOURCES: live streams / local MP4   (CAM-01 .. CAM-04)         │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  🖥️  EDGE SERVER   (FastAPI / Uvicorn on 0.0.0.0 and ::, port 8000)      │
│                                                                          │
│  api/cameras.py      ingest, telemetry, YOLO inference (worker threads) │
│                                   │                                      │
│                                   ▼                                      │
│  ai_detection/        YOLOv8 fleet (5 models) + calibration/tracking    │
│  ai_behavior/         zero_dce.py · posture_classifier.py ·             │
│                        laplacian_check.py · histogram_check.py          │
│                                   │                                      │
│                                   ▼                                      │
│  rule_engine/geofence_check.py    ray-casting geofence breach test      │
│                        │                                                 │
│      ┌─────────────────┼──────────────────────┐                         │
│      ▼                 ▼                      ▼                         │
│  🔐 crypto_ledger/   📡 ws_alerts.py       🗄️  storage/cleaner.py        │
│  merkle_ledger.py    /ws/alerts, via       FIFO 92% → 80%               │
│  SHA-256 chain +     run_coroutine_        media_tombstone_log          │
│  hourly Merkle root  threadsafe                                         │
│                                                                          │
│  SQLite trinetra.db │ storage/clips, thumbnails, exports                │
│  search_service/ (Hindi/Hinglish)   export_service/ (signed ZIP)        │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
                    NEW_ALERT (JSON) over WebSocket /ws/alerts
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  🧑‍✈️ OPERATOR DESK   backend/app/static/index.html (vanilla HTML/CSS/JS) │
│  4-camera 2×2 matrix │ polygon geofence canvas │ lux + NVG filter        │
│  threat modal + siren │ audit ledger │ search │ forensic ZIP export      │
└──────────────────────────────────────────────────────────────────────────┘
```

<br>

---

## 📁 Repository / File Structure

| Path | Layer | Role |
|---|:---:|---|
| `run_local.py` | 🚀 Bootstrap | Initialises SQLite (`trinetra.db`), creates the media hierarchy, launches Uvicorn dual-stack (`0.0.0.0` + `::`) on port `8000` |
| `backend/app/main.py` | ⚙️ Backend | FastAPI entry point — CORS, static mounts, routers, WebSocket lifecycle |
| `backend/app/api/cameras.py` | ⚙️ Backend / 🧠 AI | Video ingest + inference coordinator (live streams, local MP4, telemetry, frame extraction) |
| `backend/app/api/ws_alerts.py` | ⚙️ Backend | WebSocket broadcaster at `/ws/alerts`; bridges worker threads to async consumers |
| `backend/app/rule_engine/geofence_check.py` | 📐 Rules | Spatial evaluation of geofence breaches |
| `backend/app/crypto_ledger/merkle_ledger.py` | 🔐 Integrity | SHA-256 hash-chained audit ledger with hourly Merkle roots |
| `backend/app/search_service/` | 🔍 Search | Local vernacular/semantic query resolution (Hindi/Hinglish) |
| `backend/app/storage/cleaner.py` | 🗄️ Storage | FIFO watchdog — prune above 92%, stop below 80%, logs to `media_tombstone_log` |
| `backend/app/export_service/` | 📦 Export | Signed forensic ZIP builder (clip + thumbnail + `evidence_metadata.json`) |
| `backend/app/static/index.html` | 🖥️ Frontend | Monolithic operator desk — vanilla HTML5/CSS3/ES6+, no framework, no CDN |
| `ai_behavior/` | 🧠 AI | `zero_dce.py` · `posture_classifier.py` · `laplacian_check.py` · `histogram_check.py` |
| `ai_detection/` | 🧠 AI | Ultralytics YOLOv8 wrapper — custom confidence calibration + tracking filters |

**📂 Created automatically at start-up:**

```
trinetra.db                  → local SQLite database
storage/clips/                → incident video clips (FIFO-pruned unless flagged)
storage/thumbnails/           → event thumbnails
storage/exports/              → generated forensic evidence packages
data/known_suspects/          → local reference directory
```

<br>

---

## 🧠 AI Model Fleet

Five specialised YOLOv8 models, **~5.9 MB each (~29.6 MB total)** — the whole fleet runs on CPU-only edge hardware.

| Model | Role | Size | Classes / Focus | Training Data |
|---|---|:---:|---|---|
| 🎯 `yolov8_custom.pt` | **Active fleet primary** | 5.91 MB | human, worker, vehicle, large_backpack — **~0.94+ mAP@0.5** | Fine-tuned on elevated CCTV footage |
| 🛰️ `yolov8_visdrone.pt` | Wide-area overhead | 5.92 MB | Distant/tiny targets at steep angles | VisDrone2019-DET (6,471 images) |
| 🔫 `yolov8_weapon.pt` | Threat sentry | 5.92 MB | gun (firearms); melee (knives, machetes, clubs) | Specialised weapon datasets |
| 🚙 `yolov8_patrol.pt` | Vehicle checkpoint | 5.92 MB | jeep, suv, sedan, semi-truck, truck, van | Perimeter-transport vehicle data |
| 🌙 `yolov8_night.pt` | Thermal / low-light | 5.91 MB | Human detection in near-zero illumination | LLVIP (paired visible-infrared) |

> ℹ️ Per-model accuracy is reported only for `yolov8_custom.pt`. Training hyper-parameters (epochs, image size, augmentation, LR schedule) are not published in this repository.

<br>

---

## 🧮 Core Algorithms

<details open>
<summary><b>📐 Geofence breach test — ray casting</b></summary>
<br>

Operators draw arbitrary polygons `P = {(x₁,y₁), …, (xₙ,yₙ)}`. For each detection's ground-contact anchor `(x₀, y₀)`, a horizontal ray is cast toward `+X∞` and edge crossings are counted per the Jordan curve theorem:

```
Crossings = Σ 1[ (yᵢ > y₀) ≠ (yᵢ₊₁ > y₀)  AND  x₀ < (xᵢ₊₁-xᵢ)(y₀-yᵢ)/(yᵢ₊₁-yᵢ) + xᵢ ]

inside(x₀, y₀, P)  ⟺  Crossings mod 2 = 1
```

An odd crossing count → the anchor is inside the polygon → **active breach**.

</details>

<details>
<summary><b>🎯 Coordinate-space normalisation</b></summary>
<br>

Geofences are drawn on a responsive HTML5 canvas; YOLO returns boxes in native video pixels. Canvas coordinates are converted to normalised ratios before the point-in-polygon test:

```
x̂ = xc / Wc,   ŷ = yc / Hc,   (x̂, ŷ) ∈ [0.0, 1.0]²
```

</details>

<details>
<summary><b>🚶 Posture estimation</b></summary>
<br>

```
AR = Width / Height
```

| Aspect Ratio | Classification | Meaning |
|:---:|---|---|
| `AR < 0.75` | 🚶 Upright / Standing / Running | Normal upright movement |
| `0.75 ≤ AR ≤ 1.25` | 🧎 Crouching / Suspicious Loitering | Suspicious loitering posture |
| `AR > 1.25` | 🐍 Crawling / Prone Infiltration | Typical of fence-breaching under cover of darkness |

</details>

<details>
<summary><b>🌙 Automatic night vision — lux estimation</b></summary>
<br>

```
Luxavg = (1 / W×H) · Σ Σ (0.299·R + 0.587·G + 0.114·B)
```

When `Luxavg < 55` (0–255 scale), the stream is flagged low-light and a green NVG filter is applied:
`filter: sepia(100%) hue-rotate(90deg) saturate(350%) contrast(1.4)`.

> This is a mean-luma proxy for scene brightness, not a calibrated photometric measurement.

</details>

<details>
<summary><b>📷 Camera tamper / defocus sentry</b></summary>
<br>

```
σ²∇ = Var(∇²I),   σ²∇ < τblur ⇒ tamper alert
```

The variance of the Laplacian of consecutive frames drops sharply on defocus or lens obstruction; `histogram_check.py` provides a complementary check. `τblur` is a configurable threshold (value not published here).

</details>

<br>

---

## 📊 Datasets

| Dataset | Purpose |
|---|---|
| 🛰️ VisDrone2019-DET | 6,471 overhead/oblique captures; hardens small-object detection |
| 🔫 Bigger Weapons Dataset (Roboflow) | Annotated handguns, rifles, bladed melee weapons |
| 🌙 LLVIP | Paired visible/infrared night surveillance captures |
| 🚗 CCPD2019 | License-plate benchmark for multi-angle ANPR calibration |
| 🚙 UA-DETRAC & VeRi-776 | Vehicle trajectory, colour, make/model attribute classification |
| 🏃 NTU RGB+D & HMDB-51 | Human action/gait recognition; calibrates posture transitions |
| 🐄 OpenImages V7 (animal subsets) | Distinguishes livestock from crouching humans to reduce false triggers |
| 🌧️ DAWN & ExDark | Adverse-weather / dark imagery for fog, rain, haze, low contrast robustness |

<br>

---

## 🚀 Getting Started

### Prerequisites

- ✅ Python 3.9+
- ✅ pip
- ✅ A CPU capable of real-time YOLOv8 inference (no GPU required; a GPU improves throughput)
- ✅ Sufficient local disk for video retention (1 TB NVMe recommended for continuous multi-stream capture)

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/trinetra-ibvap.git
cd trinetra-ibvap

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### ▶️ First Run

```bash
python run_local.py
```

This will:
1. 🗃️ Initialise `trinetra.db` (SQLite) if it doesn't already exist
2. 📁 Create the `storage/` and `data/known_suspects/` directory hierarchy
3. 🌐 Start Uvicorn bound to `0.0.0.0` (IPv4) **and** `::` (IPv6) on port `8000` — avoiding the `ERR_CONNECTION_REFUSED` errors seen on Windows when a browser alternates between `127.0.0.1` and `::1`

Then open:

```
http://localhost:8000
```

> 🔒 **No internet connection is required at any point** — all assets (fonts, icons, scripts) are served locally.

<br>

---

## 🖥️ Usage Guide

| Step | Action |
|:---:|---|
| 1️⃣ | **Connect cameras** — assign a live stream URL or upload a local MP4 to each of the four camera slots (CAM-01 – CAM-04) |
| 2️⃣ | **Draw a geofence** — click points directly on a camera feed to draw a polygon perimeter (convex or non-convex); close the polygon to activate it |
| 3️⃣ | **Monitor** — detections are drawn as overlays in real time; posture (standing/crouching/crawling) is shown per entity |
| 4️⃣ | **Respond to alerts** — a geofence breach triggers a threat modal + siren once per session per camera; every event lands in the audit ledger |
| 5️⃣ | **Search** — use natural English, Hindi or Hinglish queries, e.g. `laal shirt aadmi`, `safed gadi`, `andhere mein crouching` |
| 6️⃣ | **Export evidence** — select an incident and export a signed forensic ZIP (video + thumbnail + SHA-256 metadata) |
| 7️⃣ | **Low-light operation** — feeds automatically switch to a green NVG-style overlay below the brightness threshold — no manual toggle needed |

### 🗄️ Storage Behaviour

The storage watchdog runs continuously: once disk usage exceeds **92%**, it deletes the oldest **unflagged** clips (oldest first) until usage falls back below **80%**. Every deletion is recorded in `media_tombstone_log`, so removals remain auditable even after the underlying file is gone.

> ⚠️ Flag any clip you need to keep before it ages out.

<br>

---

## 🔌 API & WebSocket Reference

| Endpoint | Type | Purpose |
|---|:---:|---|
| `/ws/alerts` | 📡 WebSocket | Broadcasts `NEW_ALERT` JSON events to all connected operator clients in real time |
| `backend/app/api/cameras.py` routes | 🌐 REST | Camera ingest, telemetry, stream/file management |
| Export routes (`export_service/`) | 🌐 REST | Generates and downloads signed forensic evidence ZIPs |
| Search routes (`search_service/`) | 🌐 REST | Resolves English/Hindi/Hinglish natural-language queries locally |

> ⏱️ Alerts are throttled server-side with a **45-second suppression window per entity ID / camera** to prevent flooding the WebSocket channel and the operator UI.

<br>

---

## ✅ Testing

```bash
pytest
```

<div align="center">

**87 / 87 automated tests passing (100%)** ✔️

</div>

Coverage includes:
- 🌐 REST API endpoints
- 📐 Geofence ray-casting mathematics
- 🚶 Posture-classification bifurcation rules
- 🌙 Low-light / night-vision handling
- 🔁 Soak-load (sustained multi-stream) scenarios

<br>

---

## ⚠️ Known Limitations

| # | Limitation | Detail |
|:---:|---|---|
| 1 | 🖥️ **CPU compute ceiling** | Four full-resolution 1080p streams at 30 FPS can saturate consumer CPUs; use frame skipping (every 3rd–5th frame) to stay real-time |
| 2 | 📐 **2D planar geofencing** | Operates in 2D camera coordinates; steep terrain or foreground obstructions (wire, ditches) can cause false boundary crossings |
| 3 | 🌫️ **Extreme optical obstruction** | Tamper sentries detect defocus/lens interference but cannot recover from catastrophic degradation (heavy mud, laser dazzle, zero-visibility blizzards) |
| 4 | 💾 **Local storage horizon** | Fully offline retention is bounded by local disk — typically days to weeks of continuous multi-stream HD footage, even with FIFO pruning |

<br>

---

## 🗺️ Roadmap

| # | Enhancement | Planned Technology | Target Outcome |
|:---:|---|---|---|
| 1️⃣ | ⚡ TensorRT compilation + edge NPU deployment | TensorRT FP16/INT8 on NVIDIA Jetson Orin Nano / AGX Orin | Sub-12 ms inference across 8+ concurrent 4K streams |
| 2️⃣ | 🧑‍🤝‍🧑 Multi-camera tracking & person Re-ID | Cross-camera embeddings (OSNet / FastReID) over local ByteTrack | One continuous identity across all cameras |
| 3️⃣ | 🌡️ Sensor fusion: thermal FLIR + PTZ slew-to-cue | Dual optical/thermal cameras, Pelco-D / ONVIF PTZ control | Motorised PTZ auto-slews and locks onto a breach location |
| 4️⃣ | 📡 Decentralised BOP tactical mesh network | Offline peer-to-peer sync over LoRa / tactical UHF | Neighbouring outposts share threat intel without internet/satellite |
| 5️⃣ | 🧊 3D volumetric LiDAR / stereoscopic depth geofencing | Solid-state LiDAR point clouds + optical video | True 3D intrusion envelopes, removing 2D depth ambiguity |

<br>

---

## 📖 Glossary

<details>
<summary>Click to expand full glossary</summary>
<br>

| Term | Meaning |
|---|---|
| **BOP** | Border Outpost |
| **BSF / MHA** | Border Security Force / Ministry of Home Affairs |
| **ASGI** | Asynchronous Server Gateway Interface (Uvicorn serves FastAPI over ASGI) |
| **YOLO / mAP@0.5** | You Only Look Once detector family / mean average precision at IoU 0.5 |
| **LLVIP** | Paired visible-infrared low-light pedestrian dataset |
| **Zero-DCE** | Zero-reference deep curve estimation (low-light image enhancement) |
| **NVG** | Night-vision goggles (here, a synthetic green display filter) |
| **FIFO** | First in, first out (oldest unflagged clips removed first) |
| **Merkle root** | Single hash summarising a set of records organised as a hash tree |
| **Re-ID** | Person re-identification across cameras |
| **PTZ / ONVIF / Pelco-D** | Pan-tilt-zoom camera / open IP-camera interface standard / PTZ control protocol |
| **TensorRT** | NVIDIA inference optimisation runtime (FP16/INT8 engines) |
| **LoRa / UHF** | Long-range low-power radio / ultra-high-frequency tactical data links |
| **LiDAR** | Light detection and ranging point-cloud sensor |

</details>

<div align="center">

---

🛡️ **TRINETRA (IBVAP)** — designed for Border Outposts, forward operating bases and critical perimeter security

### *Sovereign by design: no cloud, no CDN, no compromise.*

</div>
