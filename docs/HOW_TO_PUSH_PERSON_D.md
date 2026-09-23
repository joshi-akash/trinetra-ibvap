# Instructions for Person D: How to Push Your Code

**Role:** Track D — Frontend & Tactical Dashboard Lead  
**Feature Branch:** `feature/track-d-frontend`  
**Target Branch for PR:** `main`  
**Repo URL:** `https://github.com/CodeOpia/trinetra-ibvap.git`

---

## 1. Directory Structure You Must Follow

You are responsible for creating files **strictly inside the `frontend/` folder**:

**Do NOT modify or add files to `backend/`, `ai_behavior/`, `ai_detection/`, or `infra/`.**

```
trinetra-ibvap/
└── frontend/
    ├── package.json               # Dependencies (React, Vite/Next, Tailwind, Lucide, Axios)
    ├── tsconfig.json              # TypeScript compiler settings
    ├── tailwind.config.js         # Tactical dark-theme palette (#0B0F14, #3FD0E0, #E5484D)
    ├── vite.config.ts             # Dev server & build configuration
    ├── index.html                 # Single page application entrypoint
    ├── Dockerfile                 # Multi-stage Nginx build for production
    └── src/
        ├── index.css              # Dark mode styling & scanline overlays
        ├── App.tsx                # Tactical shell with 72px left icon rail
        ├── api/
        │   └── client.ts          # Typed Axios client targeting /api/v1 (see docs/api-contract.yaml)
        ├── ws/
        │   └── alertsSocket.ts    # WebSocket client connecting to ws://localhost:8000/ws/alerts
        ├── mocks/
        │   └── mockApi.ts         # Mock fixtures matching OpenAPI contract for offline dev
        ├── components/
        │   └── PushToTalk.tsx     # Hands-free keyword spotter ("Mark False", "Sound Alarm")
        ├── screens/
        │   ├── LiveView/          # Multi-camera matrix with live bounding boxes & tamper badges
        │   ├── AlertFeed/         # Real-time alert feed with quick acknowledgement
        │   ├── AlertDetail/       # 30s video clip viewer, WebP thumbnail & forensic metadata
        │   ├── DigitalTwinMap/    # 2D tactical map with camera FOV cones & live breadcrumb tracks
        │   ├── ForensicSearch/    # Natural language & Hinglish vernacular search interface
        │   ├── GeoFenceEditor/    # Interactive polygon drawing tool on camera snapshots
        │   └── Settings/          # System health, disk threshold meter & camera calibration
        └── tests/
            └── test_ui_standalone.test.ts # Jest / Vitest component tests
```

> **Critical Seam Rules:**  
> 1. REST endpoints must strictly match the OpenAPI specification at [`docs/api-contract.yaml`](./api-contract.yaml).
> 2. Live WebSocket alerts connect to `ws://localhost:8000/ws/alerts`.
> 3. Voice PTT strictly enforces safe keyword grammar: `"Mark False"`, `"Sound Alarm"`, `"Track Target"` (never trigger actions like QRT dispatch by voice).

---

## 2. Exact Step-by-Step Terminal Commands

### Step 1: Clone or Pull the Latest `main` Branch
Open your terminal (PowerShell, Bash, or Command Prompt):
```bash
git clone https://github.com/CodeOpia/trinetra-ibvap.git
cd trinetra-ibvap
git checkout main
git pull origin main
```

### Step 2: Create Your Dedicated Feature Branch
```bash
git checkout -b feature/track-d-frontend
```

### Step 3: Place Your Code Inside `frontend/`
Put all your React/TypeScript files, components, and screens inside the `frontend/` directory following the layout above.

### Step 4: Verify Local Install and Build
Ensure your project builds without any TypeScript or bundling errors:
```bash
cd frontend
npm install
npm run build
cd ..
```
*(Ensure `dist/` or `node_modules/` are not tracked by checking git status).*

### Step 5: Check Git Status
From the root of the repository:
```bash
git status
```
*(You should see only `frontend/` as modified/untracked. Ensure `node_modules/` is ignored).*

### Step 6: Stage, Commit, and Push
```bash
git add frontend/
git commit -m "feat(frontend): implement Person D tactical dashboard, live feed, digital twin map, and alerts"
git push -u origin feature/track-d-frontend
```

### Step 7: Open a Pull Request on GitHub
1. Open [https://github.com/CodeOpia/trinetra-ibvap](https://github.com/CodeOpia/trinetra-ibvap).
2. Click the green button: **Compare & pull request** for `feature/track-d-frontend`.
3. Set the base branch to **`main`**.
4. Click **Create pull request** and merge!
