# Person B — Behavior, Low-Visibility & Hardening AI Lead

**You own:** `ai_behavior/` in the shared monorepo, including the joint
`orchestrator/` seam (built with Person A — see §6).
**You build:** posture/behavior analytics, night-time/low-light enhancement,
camera tamper/blinding detection, the frame-processing pipeline that chains
Person A's models with yours, and — critically — the threshold-tuning work that
makes the whole AI stack trustworthy rather than noisy.
**Read first:** `docs/00_INTEGRATION_CONTRACT.md` — you complete the
`FrameAnalysis` object that Person A starts (§2 of that doc); the fields
`posture`, `props`, and `frame_quality` are yours.

Hand this file, together with the integration contract, to an AI coding agent
to build from scratch. Everything it needs — stack, layout, exact outputs, and
build order — is below.

---

## 1. What You're Building, in One Paragraph

Detection tells you *what* is in a frame; you tell the system *how it's
behaving* and *how trustworthy the frame itself is*. You add posture analysis
(crouching, sprinting — the "suspicious activity" the mandate requires), you
make the system usable at night (low-light enhancement + adaptive confidence),
and you make the system honest about degraded conditions (tamper detection, and
an explicit "visual confidence lost" state rather than a silent miss). You also
own the orchestrator that chains everything into one object per frame — this
makes you the person most responsible for the AI stack's overall timing budget
and stability, not just your own models' accuracy.

---

## 2. Tech Stack (do not substitute without team sign-off)

| Component | Choice | Why |
|---|---|---|
| Pose/posture | MediaPipe (pose landmarks) | Lightweight, runs alongside YOLOv8 on the same GPU box without heavy extra load |
| Night enhancement | Zero-DCE | Matches spec; low-light frame enhancement without needing paired training data |
| Tamper detection | Laplacian variance collapse + histogram shape + reference-point drift + frame-delta collapse | No model training needed — pure CV signal processing, cheap and fast to build |
| Geo-fence math (you consume, don't own) | Shapely — owned by Person C | You don't implement polygon intersection; you supply *inputs* to Person C's Rule Engine (posture flags, tamper flags) |

Install: `pip install mediapipe opencv-python numpy scipy`. Zero-DCE has a
small PyTorch reference implementation — pull a pretrained checkpoint rather
than training from scratch (there's no time budget for that in this plan, and
none is required — see PRD §9 assumptions on pretrained models being acceptable).

---

## 3. Folder Layout (inside `ai_behavior/`)

```
ai_behavior/
├── pose_behavior/
│   ├── posture_classifier.py    # MediaPipe landmarks → crouching/sprinting/normal
│   └── prop_detection.py        # covered face, large bag — feeds off Person A's bbox crops
├── night_enhancement/
│   ├── zero_dce.py              # low-light frame enhancement
│   └── adaptive_threshold.py    # adjusts detection confidence floor based on measured low-light severity
├── tamper/
│   ├── laplacian_check.py       # blur/blinding detection via variance collapse
│   ├── histogram_check.py       # sudden histogram tightening (lens covered)
│   └── reference_drift.py       # camera moved/reangled vs calibration reference points
├── orchestrator/                 # JOINT with Person A — skeleton built by A in Week 1, extended by you Week 2–3
│   └── pipeline.py               # chains: night_enhancement (if needed) → Person A's detection stage → your posture/props → tamper check → assembles final FrameAnalysis
├── thresholds.yaml                # the config file Backend's Rule Engine reads — see integration contract §3
├── tests/
│   └── test_staged_footage.py
└── __init__.py
```

---

## 4. Output Contract (your part of `FrameAnalysis`)

You complete the object Person A starts. Per entity, you add:

```json
{
  "attributes": {
    "posture": "crouching",
    "props": ["covered_face", "large_bag"]
  }
}
```

And at the frame level (not per-entity), you own the entire `frame_quality`
block:

```json
{
  "frame_quality": {
    "low_light": true,
    "enhanced": true,
    "tamper_signal": {
      "is_tampered": false,
      "laplacian_variance": 812.4,
      "histogram_flag": false,
      "reference_point_drift": false
    }
  }
}
```

When `tamper_signal.is_tampered` flips true, that's what drives Person C's
distinct tamper-alert path (SRS FR-TMP-01/02) — separate from a normal
visibility-degradation alert. Get this distinction right; conflating "it's
foggy" with "someone covered the lens" is a real SRS acceptance criterion (PRD
§12: covering a camera lens must trigger a tamper alert distinct from normal
degraded-visibility handling).

---

## 5. Week-by-Week Build Order

### Week 1 — Foundation
- **Mon:** Pull MediaPipe and a Zero-DCE pretrained checkpoint. Smoke-test each standalone.
- **Tue:** Run each against one sample frame; log inference time (feeds the same NFR-02 latency budget Person A is measuring for their stage).
- **Wed:** Stage a night-condition test clip and a fog/dust clip if not already covered by Person A's Week 1 sourcing — coordinate so you're not duplicating effort.
- **Thu:** Work with Person A to confirm the combined 5+ model chain (their stage + yours) fits the GPU concurrency budget. This is a shared risk — surface it Thursday, not Week 4.
- **Fri:** Sign off on the frozen contracts. Start the `orchestrator/pipeline.py` skeleton together with Person A — agree on the exact call order (night enhancement must run *before* detection when needed, so Person A's models see an enhanced frame).

### Week 2 — Wire Into the Real Pipeline
- **Mon:** Feed behavior triggers (once built) into the orchestrator; get real posture data flowing.
- **Tue:** Nothing new required yet from you specifically per the base plan — use this day to get ahead on night enhancement so Week 3's tuning has real material to work with.
- **Wed:** Continue integration support — height estimation calibration (Person A) may surface orchestrator issues you need to fix.
- **Thu:** Continue supporting FRS/ANPR wiring from the orchestrator side — make sure your pipeline correctly skips posture/props for vehicle entities and skips plate/face fields for human entities where not applicable.
- **Fri:** Sanity-check the full chain output once against the `FrameAnalysis` schema.

**Friday checkpoint (shared with Person A):** a person walking through a test
frame produces a complete `FrameAnalysis` object — detection fields from Person
A, behavior/quality fields from you — flowing into a real database row via
Person C's ingestion path.

### Week 3 — Your Core Build Weeks
- **Mon:** Add MediaPipe pose → crouching/sprinting triggers (FR-BEH-01). Feed as a second active-alert condition into Person C's Rule Engine (you supply the trigger; Person C wires the bifurcation logic).
- **Tue:** Add Zero-DCE night enhancement + adaptive confidence thresholding on a real night test clip.
- **Wed:** Tune per-class confidence thresholds jointly with Person A against Week 1's staged test clips; log false-positive/negative spot-checks.
- **Thu:** Begin tamper-detection signal (Laplacian variance collapse) if on schedule — this is explicitly a stretch task per the source Execution Plan (P1, Week 5 fallback if behind). **If Week 3 core behavior/night work isn't done, skip tamper entirely this week and pick it up in Week 5** — do not let it block the Week 4 integration milestone.
- **Fri:** Full pipeline dry-run: one clip exercising every detection type (yours and Person A's) end-to-end through the orchestrator.

**Friday checkpoint:** every AI capability in MVP scope — yours and Person A's —
has produced at least one real, verifiable log row through the real pipeline.

### Week 4 — Integration (joint week, see integration contract §5)
Remove mocks, help debug the full loop. You're the most likely person to be
needed for timing/latency issues since you own the orchestrator — budget your
week accordingly, expect to be pulled into other people's debugging sessions.

### Week 5 — Hardening
- **Mon:** False-positive/negative spot-check across day/night/degraded-visibility, jointly with Person A. Tune thresholds via `thresholds.yaml`, not architecture.
- **Tue:** Finish tamper-detection signal if not already done in Week 3 — distinct alert type from normal visibility degradation. This is the last point at which it's safe to cut it if behind; if it's not working by Wednesday, cut it (per the source plan's own risk watchlist) rather than let it eat time from the P0 hardening work everyone needs.
- **Wed–Thu:** Bug triage and fix cycle with the team.
- **Fri:** Support the multi-hour soak run — watch for any drift in your adaptive-threshold logic over long runtimes.

### Week 6 — Demo Readiness
Support runbook/demo-script work; be available for both dry-runs, especially the
WAN-disconnected segment (verify your tamper/low-light logic behaves identically
with or without connectivity — it should never reference the network at all).

---

## 6. Coordination Points

- **Person A:** joint ownership of `orchestrator/pipeline.py` — you extend the
  skeleton they build in Week 1. Agree explicitly on call order (enhancement
  before detection) and on how partial results are merged into one object.
- **Person C:** you hand them `thresholds.yaml` (the tunable trigger definitions)
  rather than hardcoding thresholds into your models — this is what lets Week 3/5
  tuning be a config change, not a redeploy. You also hand them the tamper
  signal that drives their distinct tamper-alert path.
- **Person D:** no direct interface, same as Person A — if something looks wrong
  in the dashboard's alert feed (e.g., wrong alert type shown), check your
  `frame_quality`/`posture`/`props` output against the contract before assuming
  it's a frontend bug.

---

## 7. Definition of Done for Your Stage

1. Tested with WAN physically disconnected — your code never touches the network under any condition.
2. Output matches the `FrameAnalysis` contract exactly, including the `frame_quality` block.
3. Has a scripted test with a stated expected outcome for at least: normal posture, suspicious posture, night footage, and a simulated tamper event (lens covered).
4. The tamper-alert path is demonstrably distinct from the normal low-visibility path — not the same trigger reused.
5. A second person (ideally Person A) looked at your output before you mark it done.

---

## 8. Prompt You Can Hand an AI Coding Agent

> Build the `ai_behavior/` module of the TRINETRA monorepo described in
> `docs/00_INTEGRATION_CONTRACT.md`. Implement MediaPipe-based posture
> classification (crouching/sprinting/normal) and prop detection (covered face,
> large bag), Zero-DCE-based low-light enhancement with adaptive confidence
> thresholding, and camera tamper/blinding detection using Laplacian variance
> collapse, histogram shape change, and calibration reference-point drift —
> producing a distinct tamper signal separate from ordinary low-visibility
> conditions. Also build `orchestrator/pipeline.py`, which calls low-light
> enhancement first (when needed), then the detection/recognition stage from
> `ai_detection/`, then your posture/prop/tamper stages, and assembles the
> complete `FrameAnalysis` object exactly matching the schema in the
> integration contract. Externalize every trigger threshold (posture
> classification cutoffs, tamper variance thresholds, confidence floors) into
> `thresholds.yaml` rather than hardcoding them. Do not implement networking,
> database access, or UI. Write unit tests against sample frames in
> `test_footage/`, including at least one simulated tamper scenario.
