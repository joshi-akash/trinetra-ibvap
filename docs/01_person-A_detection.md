# Person A — Detection & Recognition AI Lead

**You own:** `ai_detection/` in the shared monorepo.
**You build:** human/vehicle detection, pedestrian soft-biometrics (PAR), height
estimation, facial recognition (FRS), and automatic number-plate recognition (ANPR).
**Read first:** `docs/00_INTEGRATION_CONTRACT.md` — the `FrameAnalysis` schema in
its §2 is the exact shape your code must output. Everything you build feeds into
that object; Backend (Person C) and Frontend (Person D) never call your models
directly, they only ever see `FrameAnalysis`.

This file is written so you can hand it, together with the integration contract,
directly to an AI coding agent (Antigravity or similar) and say "build this from
scratch." It contains everything the agent needs: stack, folder layout, exact
model choices, output shapes, and a week-by-week build order.

---

## 1. What You're Building, in One Paragraph

Every frame that comes off a camera needs to answer: is there a human, vehicle,
or animal in it; what color are they wearing; how tall are they; does their face
match a known suspect; if it's a vehicle, what does the plate say. You build the
five model stages that answer these questions and package each answer into the
`entities[]` array of a `FrameAnalysis` object (contract in §2 of the integration
doc). You do not touch the database, the API, the rule engine, or the dashboard —
your only interface to the rest of the system is that one JSON shape.

---

## 2. Tech Stack (do not substitute without team sign-off)

| Component | Choice | Why |
|---|---|---|
| Detection | YOLOv8 (pretrained), exported to ONNX/TensorRT for inference | Matches spec; TensorRT export needed for GPU throughput on the VA server |
| Pose (feeds Person B, not you) | MediaPipe | Person B owns this — mentioned here only so you know not to duplicate it |
| Face recognition | InsightFace: RetinaFace (detection) + ArcFace (embedding) | Matches spec; local known-suspect database, no cloud API |
| ANPR | PaddleOCR (primary), EasyOCR (fallback) | Matches spec; tuned for Indian plate formats |
| Low-light enhancement (feeds you, not built by you) | Zero-DCE | Person B owns this — your models must accept an already-enhanced frame from the orchestrator when `frame_quality.enhanced = true` |
| Color/PAR | HSV clustering (OpenCV) on the person bounding box | Lightweight, CPU-friendly, no separate model needed |
| Height | Calibrated perspective geometry using `camera_registry.calibration_reference_points` | Needs one admin-supplied calibration set per camera — coordinate with Person C on the calibration endpoint |
| Runtime | PyTorch for model loading/export; ONNX Runtime or TensorRT for inference | GPU inference throughput |

Install everything with `pip install ultralytics insightface paddleocr paddlepaddle-gpu opencv-python onnxruntime-gpu` (adjust CUDA version to the VA server's actual driver).

---

## 3. Folder Layout (inside `ai_detection/`)

```
ai_detection/
├── detection/
│   ├── yolov8_detector.py       # loads pretrained weights, runs inference, returns raw boxes+classes
│   └── weights/                 # gitignored — see models/download_weights.sh at repo root
├── par/
│   └── clothing_color.py        # HSV clustering on person bbox → upper_color, lower_color
│   └── gender_estimation.py     # confidence-gated: below threshold → "neutral", never guessed
├── height/
│   └── perspective_height.py    # uses camera_registry.calibration_reference_points
├── frs/
│   ├── face_matcher.py          # RetinaFace detect + ArcFace embed + cosine similarity vs known-suspect table
│   └── known_suspects.py        # loads/queries the local suspect embedding store (seeded via Backend's DB, see §6)
├── anpr/
│   ├── plate_detector.py        # YOLOv8 plate-crop
│   └── plate_ocr.py             # PaddleOCR primary, EasyOCR fallback
├── tests/
│   └── test_staged_footage.py   # smoke tests against test_footage/
└── __init__.py                  # exposes run_detection_stage(frame, camera_id) -> partial FrameAnalysis
```

Your package's single public function, `run_detection_stage()`, is what Person
B's orchestrator (`ai_behavior/orchestrator/`) calls. It takes a frame + camera_id
and returns the detection/recognition portion of the `entities[]` array — Person
B's orchestrator then adds posture, props, and frame-quality fields before handing
the completed `FrameAnalysis` object to Backend.

---

## 4. Output Contract (your part of `FrameAnalysis.entities[]`)

For each detected entity, you are responsible for populating:

```json
{
  "track_id": "temp-uuid-for-this-frame",
  "entity_type": "human",
  "bbox": [412, 88, 560, 410],
  "confidence": 0.91,
  "attributes": {
    "upper_color": "blue",
    "lower_color": "black",
    "height_cm": 172.4,
    "gender": "neutral",
    "plate_text": null,
    "face_match": { "suspect_id": "SUSP-002", "confidence": 0.87 }
  }
}
```

Fields you leave `null` (Person B fills them): `posture`, `props`,
`frame_quality`. Vehicles get `plate_text` populated and `face_match`/`gender`/
`height_cm` left `null`. Never fabricate a value to fill a field you're not
confident about — a `null` is correct; a guess is a bug (this is a direct SRS
requirement: FR-PAR-02 confidence-gated gender, FR-FRS-03 no fabricated matches).

---

## 5. Week-by-Week Build Order

### Week 1 — Foundation
- **Mon:** Pull YOLOv8, InsightFace, PaddleOCR pretrained weights into `models/`. Smoke-test each in a standalone script (one image in, structured output out) — don't wait for the pipeline to exist.
- **Tue:** Run each model against one sample frame; log inference time per model. This feeds the NFR-02 latency budget the whole team needs to know by Friday.
- **Wed:** Source/stage test RTSP footage if no live camera is available yet — day, night, and at least one clip with a clear geo-fence-style crossing (coordinate with Person C, who needs this for ingestion testing too).
- **Thu:** Confirm the GPU can run all 5 of your models concurrently without OOM at the target resolution/camera count. Flag immediately if it can't — this is a live risk item the whole team needs to know about by Friday, not discovered in Week 4.
- **Fri:** Sign off on the frozen API contract and `FrameAnalysis` schema alongside the other 3. Build `run_detection_stage()`'s skeleton — even if it returns stub data, Person B's orchestrator needs something to call.

### Week 2 — Wire Into the Real Pipeline
- **Mon:** Wire YOLOv8 into the orchestrator skeleton (built with Person B); get real bounding boxes flowing per frame from Person C's RTSP ingestion service.
- **Tue:** Add PAR — clothing color HSV clustering, confidence-gated gender.
- **Wed:** Add height estimation. This needs one calibration reference set — coordinate directly with Person C on `camera_registry.calibration_reference_points` and the `/api/cameras/{id}/calibrate` endpoint.
- **Thu:** Wire FRS (InsightFace) with a manually-seeded 2–3 person known-suspect table (ask Person C how suspect embeddings get into Postgres — likely a `bytea`/vector column, admin-only enrollment).
- **Fri:** Wire ANPR (YOLOv8 plate-crop + PaddleOCR) against staged plate footage.

**Friday checkpoint:** a person walking through a test frame produces every
detection/recognition field your stage owns, flowing into a real `entity_log`
row via Person C's ingestion path — even if the dashboard doesn't show it yet.

### Week 3 — Tuning
- **Mon–Wed:** Tune per-class confidence thresholds against Week 1's staged test clips (day, night, degraded visibility). Log false-positive/false-negative spot-checks — this is real data the team needs for Week 5 hardening, start the log now, don't wait.
- Coordinate with Person B: your models need to work correctly on **both** raw and Zero-DCE-enhanced frames (Person B tells you which flag in `frame_quality.enhanced` to check).
- **Fri:** Full dry-run of your stage alone: one clip exercising detection, PAR, height, FRS, and ANPR end-to-end, output validated against the `FrameAnalysis` schema exactly.

### Week 4 — Integration (joint week, see integration contract §5)
Remove all mocks, help debug the full end-to-end loop with the other 3. Your job
this week is fixing anything in your stage that breaks under real, continuous,
WAN-disconnected operation — not building new features.

### Week 5 — Hardening
- False-positive/negative spot-check across day/night/degraded-visibility test
  clips; tune thresholds, not architecture (architecture changes this late are
  how demos break).
- If ahead of schedule, help Person B with tamper-detection tuning (P2/stretch —
  do not let it eat time from your own P0 threshold work).

### Week 6 — Demo Readiness
Support the runbook/demo-script work; be available for the non-author standup
test and both dry-runs to fix anything that surfaces in your stage live.

---

## 6. Coordination Points (who you talk to, and about what)

- **Person B:** you jointly own `ai_behavior/orchestrator/` (see integration
  contract §3) — you build its skeleton in Week 1 since detection runs first in
  the chain; Person B extends it as their models come online in Week 2–3.
- **Person C:** you need (a) the calibration endpoint for height estimation, (b)
  wherever known-suspect embeddings get stored/enrolled, (c) the RTSP frame
  source your `run_detection_stage()` gets called with. Do not build your own
  RTSP handling — that's Backend's ingestion service, you only receive frames.
- **Person D:** no direct interface — the dashboard only ever sees your output
  via the database, through Person C. If Person D reports something looks wrong
  in the UI, the first place to check is whether your `FrameAnalysis` output
  matches the contract exactly (field names, `null` vs empty string, etc.).

---

## 7. Definition of Done for Your Stage

Check every one of these before calling a task complete:
1. Tested against footage where the WAN is irrelevant (your stage never talks to the network at all — if it ever does, that's a bug).
2. Output matches the `FrameAnalysis.entities[]` contract exactly — no extra fields, no missing fields, `null` used correctly.
3. Has a scripted test with a stated expected outcome (not "looks right").
4. Every alert-relevant field traces to a specific SRS requirement (FR-DET, FR-PAR, FR-HGT, FR-FRS, FR-ANPR sections).
5. A second person (ideally Person B, since you share the orchestrator) looked at your output before you mark it done.

---

## 8. Prompt You Can Hand an AI Coding Agent

> Build the `ai_detection/` module of the TRINETRA monorepo described in
> `docs/00_INTEGRATION_CONTRACT.md`. Implement YOLOv8-based human/vehicle
> detection, HSV-based clothing color + confidence-gated gender estimation,
> perspective-geometry height estimation using camera calibration reference
> points, InsightFace-based facial recognition against a local known-suspect
> store, and PaddleOCR-based ANPR with EasyOCR fallback. Expose a single public
> function `run_detection_stage(frame, camera_id) -> list[Entity]` matching the
> `entities[]` shape in the integration contract's `FrameAnalysis` schema
> exactly. Never fabricate a field you're not confident about — return `null`
> instead. Do not implement networking, database access, or UI — your only
> output is the in-memory structured object described in the contract. Write
> unit tests against sample frames in `test_footage/` for every sub-module.
