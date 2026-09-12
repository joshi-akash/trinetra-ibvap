# Instructions for Person A: How to Push Your Code

**Role:** Track A — Detection, Tracking & Biometrics Lead  
**Feature Branch:** `feature/track-a-detection`  
**Target Branch for PR:** `main`  
**Repo URL:** `https://github.com/joshi-akash/trinetra-ibvap.git`

---

## 1. Directory Structure You Must Follow

You are responsible for creating files **strictly inside these three folders**:
1. `ai_detection/` (All your AI logic and tests)
2. `models/` (Download weights script and `.gitignore`)
3. `test_footage/` (Sample test clips and documentation)

**Do NOT modify or add files to `backend/`, `ai_behavior/`, or `infra/`.**

```
trinetra-ibvap/
├── ai_detection/
│   ├── __init__.py                # MUST expose: run_detection_stage(frame, camera_id)
│   ├── detection/
│   │   ├── yolov8_detector.py     # YOLOv8 bounding box & foot-point inference
│   │   └── tracker.py             # ByteTrack multi-object tracking
│   ├── par/
│   │   ├── clothing_color.py      # HSV torso & leg color clustering
│   │   └── gender_estimation.py   # Soft-biometric gender classifier
│   ├── height/
│   │   └── perspective_height.py  # Ground homography physical height estimation
│   ├── frs/
│   │   ├── face_matcher.py        # Cosine similarity face embedding matcher
│   │   └── known_suspects.py      # Local suspect database loader
│   ├── anpr/
│   │   ├── plate_detector.py      # Vehicle license plate crop locator
│   │   └── plate_ocr.py           # PaddleOCR / EasyOCR with Indian regex
│   ├── training/
│   │   ├── train_yolo_border.py   # Fine-tuning script
│   │   └── export_tensorrt.py     # FP16 TensorRT engine exporter
│   └── tests/
│       └── test_detection_standalone.py # Standalone pytest suite
├── models/
│   ├── .gitignore                 # Must ignore *.pt, *.pth, *.onnx, *.engine
│   └── download_weights.sh        # Shell script to pull YOLO/Zero-DCE weights
└── test_footage/
    ├── .gitkeep                   # Preserves folder in Git
    └── README.md                  # Test clips format specification
```

> **Critical Seam Rule:**  
> `ai_detection/__init__.py` must expose:
> ```python
> def run_detection_stage(frame: np.ndarray, camera_id: str) -> list[dict]:
>     ...
> ```
> Every entity dictionary returned must conform to `docs/frame-analysis-schema.json`.

---

## 2. Exact Step-by-Step Terminal Commands

### Step 1: Clone or Pull the Latest `main` Branch
Open your terminal (PowerShell, Bash, or Command Prompt):
```bash
git clone https://github.com/joshi-akash/trinetra-ibvap.git
cd trinetra-ibvap
git checkout main
git pull origin main
```

### Step 2: Create Your Dedicated Feature Branch
```bash
git checkout -b feature/track-a-detection
```

### Step 3: Copy Your Code into Place
Copy your files into `ai_detection/`, `models/`, and `test_footage/` following the layout above.

### Step 4: Verify Models Are Not Staged (Large Binaries)
Ensure large weights (`.pt`, `.onnx`, `.engine`) are not being tracked:
```bash
git status
```
*(You should see untracked files in `ai_detection/`, `models/`, and `test_footage/`, but NO large model weights).*

### Step 5: Run Your Standalone Test Suite
```bash
python -m pytest ai_detection/tests/ -v
```
*(Make sure your tests pass before pushing).*

### Step 6: Stage, Commit, and Push
```bash
git add ai_detection/ models/ test_footage/
git commit -m "feat(detection): implement Person A YOLO detection, tracking, biometrics, FRS, and ANPR"
git push -u origin feature/track-a-detection
```

### Step 7: Open a Pull Request on GitHub
1. Open [https://github.com/joshi-akash/trinetra-ibvap](https://github.com/joshi-akash/trinetra-ibvap).
2. Click the green button: **Compare & pull request** for `feature/track-a-detection`.
3. Set the base branch to **`main`**.
4. Click **Create pull request** and merge!
