# TRINETRA — Integration Contract (Read by all 4 people, owned by no one alone)

This is the single source of truth that keeps four independently-built slices of
TRINETRA combinable into **one working system**. Every person-specific file
(`01_person-A...md` through `04_person-D...md`) links back here for anything
that crosses a role boundary. **If your work touches anything in this file,
you must get a same-day sign-off from whoever else it affects before changing it** —
this is the one rule that prevents "it worked on my machine" at integration time.

Give this file to your AI coding agent (Antigravity, Claude Code, Cursor, etc.)
alongside your own person-file at the start of every session. Your person-file
tells the agent *what to build*; this file tells it *the exact shape everyone
else expects it to plug into*.

---

## 1. One Repo, Four Folders

Everyone works in **one monorepo**, not four separate repos. This is non-negotiable —
separate repos are how integration debt hides until Week 4.

```
trinetra/
├── docker-compose.yml              # Person C owns; everyone can run `docker compose up`
├── docs/
│   ├── 00_INTEGRATION_CONTRACT.md  # this file
│   ├── api-contract.yaml           # OpenAPI spec — Person C drafts, all 4 sign off
│   └── frame-analysis-schema.json  # AI→Backend contract — Person A+B draft, Person C signs off
├── infra/                          # Person C — Docker configs, Postgres init, Mosquitto config, systemd units
├── backend/                        # Person C — FastAPI app, migrations, ingestion, rule engine, alert manager, export, search
│   └── app/
│       ├── api/                    # REST + WebSocket routes
│       ├── ingestion/              # RTSP + ring buffer
│       ├── rule_engine/            # geo-fence check, passive/active bifurcation, behavior/tamper wiring
│       ├── alert_manager/          # alert package assembly
│       ├── export_service/
│       ├── search_service/
│       └── auth/
├── ai_detection/                   # Person A — YOLOv8, PAR, height, FRS, ANPR
│   ├── detection/
│   ├── par/
│   ├── height/
│   ├── frs/
│   └── anpr/
├── ai_behavior/                    # Person B — pose/behavior, night enhancement, tamper, pipeline orchestrator
│   ├── pose_behavior/
│   ├── night_enhancement/
│   ├── tamper/
│   └── orchestrator/               # chains ai_detection + ai_behavior into one FrameAnalysis object — JOINT OWNERSHIP, see §3
├── frontend/                       # Person D — React + TS + Tailwind dashboard
│   └── src/
├── models/                         # shared, gitignored pretrained weights + a `download_weights.sh` script
└── test_footage/                   # shared staged clips (day/night/fog/geo-fence-crossing) everyone tests against
```

Everyone clones the same repo, works on their own folder(s), and opens PRs. **No
one edits another person's folder without a message first** — except the two
joint-ownership seams below, which exist precisely so no folder is silently
abandoned.

---

## 2. The Two Contracts That Actually Matter

Everything else in the system is negotiable at the code level. These two JSON/API
shapes are not — they are what makes four independent builds into one product.

### Contract 1 — `FrameAnalysis` (AI side → Backend side)

This is the object the AI pipeline (Person A's detection/recognition modules,
chained with Person B's behavior/night/tamper modules) hands to the Backend's
ingestion service, once per processed frame. Backend's Rule Engine consumes this
directly — it never calls a model itself.

```json
{
  "camera_id": "CAM-04",
  "timestamp": "2026-09-11T10:15:32.120Z",
  "frame_ref": "ring-buffer-pointer-or-frame-id",
  "frame_quality": {
    "low_light": true,
    "enhanced": true,
    "tamper_signal": {
      "is_tampered": false,
      "laplacian_variance": 812.4,
      "histogram_flag": false,
      "reference_point_drift": false
    }
  },
  "entities": [
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
        "face_match": { "suspect_id": "SUSP-002", "confidence": 0.87 },
        "posture": "crouching",
        "props": ["covered_face"]
      },
      "location": { "lat": 29.9457, "lon": 78.1642 }
    }
  ]
}
```

Rules for this contract:
- `entity_type` is one of `human | vehicle | animal` — matches `entity_log.entity_type` in the DB (§4).
- Any field the model genuinely can't determine is `null`, never a guessed default — this is what lets `gender` show as "Neutral" instead of a fabricated guess (SRS FR-PAR-02).
- `face_match` and `plate_text` are only populated for entities where the relevant model ran (a human gets `face_match`, never `plate_text`; a vehicle gets `plate_text`, never `face_match` or `posture`).
- This object is produced **in-process** (a Python function call from Backend's ingestion loop into the AI pipeline package, or a local gRPC call if the pipeline runs as a separate process for GPU isolation) — never over the network, and never leaves the VA server.
- Person A and Person B each own the sub-fields their own models produce; the **shape** of the object itself is fixed by this contract and changing it requires sign-off from Person C (who consumes it) too.

### Contract 2 — REST + WebSocket API (Backend side → Frontend side)

Full contract lives in `docs/api-contract.yaml` (OpenAPI), frozen at the end of
Week 1 (§5). Representative surface — see Person C's file for the complete list
and Person D's file for how each endpoint maps to a screen:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/auth/login` | POST | Issue JWT |
| `/api/cameras` | GET | List cameras + live trust/status |
| `/api/cameras/{id}/calibrate` | POST | Submit ground-plane reference points |
| `/api/geofence/{camera_id}` | PUT | Update geo-fence polygon |
| `/api/entities` | GET | Query `entity_log` (time, color, type, alert-only filters) |
| `/api/entities/{id}/false-flag` | POST | Mark detection as false positive |
| `/api/entities/{id}/export` | POST | Build + hand off HQ export package |
| `/api/search` | POST | Structured/synonym forensic search |
| `/api/alerts/active` | GET | Current unacknowledged alerts |
| `/api/alerts/{id}/acknowledge` | POST | Acknowledge alert |
| `/ws/alerts` | WS | Live alert push, live bounding boxes, trust-score updates |

Rules for this contract:
- Drafted by Person C, reviewed by Person D (who knows what the UI needs) and flagged against by Person A/B (who knows what data is actually producible), **frozen in writing** (commit the YAML) by end of Week 1.
- After freeze, **any change requires same-day 4-way sign-off**. This is the single highest-leverage rule in the whole plan — a stale mock is how integration debt hides until the final week.
- Person D builds against this contract with mocked responses from Day 1 — never waits for Person C's real implementation to start UI work.

---

## 3. Joint-Ownership Seams (so nothing falls through the crack)

Two things sit exactly on a role boundary. Both are called out explicitly in
both adjacent person-files, but the rule is stated once, here, so it isn't lost:

1. **`ai_behavior/orchestrator/`** — the code that chains Person A's models
   (detection → PAR → height → FRS → ANPR) with Person B's models (pose/behavior →
   night enhancement → tamper) into one `FrameAnalysis` object per frame.
   Person A builds the skeleton in Week 1 (since detection runs first in the
   chain and everything else attaches after it); Person B extends it in Week 2–3
   as their models come online. Both test against it; neither owns it alone.
2. **Geo-fence + behavior "conditions" vs "plumbing"** — Backend (Person C)
   writes the actual Shapely polygon-intersection code and the passive/active
   bifurcation logic, but the *thresholds and trigger definitions* (which
   posture counts as "suspicious," what confidence floor gates a face match,
   what Laplacian-variance delta means "tampered") are Person A/B's call,
   supplied as a config file (`ai_behavior/thresholds.yaml`) that Backend reads
   rather than hardcodes. This keeps tuning (Week 3, Week 5) a one-file change,
   not a backend redeploy.

---

## 4. Database Schema (Person C builds it; everyone else's output must fit it)

```sql
CREATE TABLE entity_log (
  id                UUID PRIMARY KEY,
  camera_id         TEXT REFERENCES camera_registry(camera_id),
  timestamp         TIMESTAMPTZ NOT NULL,
  entity_type       TEXT,             -- human | vehicle | animal
  upper_color       TEXT,
  lower_color       TEXT,
  height_cm         FLOAT,
  gender            TEXT,             -- male | female | neutral
  plate_text        TEXT,
  location          GEOGRAPHY(Point),
  trajectory_id     UUID,
  is_alert          BOOLEAN,
  alert_type        TEXT,             -- geo_fence | behavior | tamper | correlated
  confidence_score  FLOAT,
  clip_path         TEXT,
  retention_tier    TEXT,             -- passive | protected
  deleted_manually  BOOLEAN DEFAULT FALSE,
  deletion_audit    JSONB
);

CREATE TABLE camera_registry (
  camera_id                     TEXT PRIMARY KEY,
  location                      GEOGRAPHY(Point),
  fov_polygon                   GEOGRAPHY(Polygon),
  geo_fence_polygon             GEOGRAPHY(Polygon),
  calibration_reference_points  JSONB,
  trust_score                   FLOAT,
  last_tamper_check             TIMESTAMPTZ
);

CREATE TABLE false_flag_log (
  id                       UUID PRIMARY KEY,
  entity_log_id            UUID REFERENCES entity_log(id),
  marked_by                TEXT,
  marked_at                TIMESTAMPTZ,
  moved_to_hard_negatives  BOOLEAN
);

CREATE TABLE users (
  id            UUID PRIMARY KEY,
  username      TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role          TEXT NOT NULL,        -- commander | operator | admin
  created_at    TIMESTAMPTZ DEFAULT now(),
  active        BOOLEAN DEFAULT TRUE
);

CREATE TABLE export_log (
  id            UUID PRIMARY KEY,
  entity_log_id UUID REFERENCES entity_log(id),
  exported_by   UUID REFERENCES users(id),
  exported_at   TIMESTAMPTZ,
  payload_hash  TEXT,
  channel       TEXT
);

CREATE TABLE audit_log (
  id          UUID PRIMARY KEY,
  user_id     UUID REFERENCES users(id),
  action      TEXT,
  target_id   UUID,
  timestamp   TIMESTAMPTZ DEFAULT now(),
  details     JSONB
);
```

`entity_type` values, `gender` values, and every field name here must match the
`FrameAnalysis` contract in §2 exactly — a mismatch here is the #1 cause of
"the AI works, the API works, but nothing shows up in the dashboard."

---

## 5. Milestones Everyone Signs Off On Together

| Milestone | What must be true | Who verifies |
|---|---|---|
| **M0/M1 (end Week 1)** | `docker compose up` gives an empty-but-running stack; all pretrained models load once each; API contract + FrameAnalysis contract both frozen and committed | All 4 |
| **M2/M3 (end Week 2)** | A real detection produces a real `entity_log` row via the real pipeline (even if not yet visible in UI); all MVP screens exist and navigate on mock data | Person A/B/C confirm the row; Person D confirms navigation |
| **Week 3 exit** | Every MVP AI capability produces a real, verifiable log row; Live View + Alert Feed + Search are wired to real data (not mocks) | All 4 |
| **M4 (end Week 4) — the big one** | (1) a live geo-fence breach produces a full alert package, (2) forensic search finds it, (3) export produces a valid local package, (4) all of the above still work with WAN physically disconnected | All 4, independently — not assumed from component tests |
| **Week 5 exit** | Zero known crash/data-loss bug; multi-hour unattended soak run completed clean | All 4 |
| **M6 (end Week 6)** | Someone who didn't build the system can stand it up from the runbook alone; two consecutive successful demo dry-runs, one fully WAN-disconnected; fallback plan tested, not just documented | All 4 |

**Do not move to the next milestone until the current one is demonstrated live**,
not inferred from individual component tests passing. This is the single
highest-leverage discipline in the whole execution plan.

---

## 6. Daily Rituals (all 4, all 6 weeks)

- **15-minute standup**, same time daily: what shipped yesterday, what's today, what's blocking. Anything blocking that touches either contract in §2 gets fixed same-day.
- **Friday checkpoint** (~30 min): walk through that week's exit line in §5 as a group, live — not a status report.
- Any change to `api-contract.yaml` or the `FrameAnalysis` schema after Week 1 needs same-day 4-way sign-off before it's merged.

---

## 7. What "Sovereignty" Means for Every Single Piece of Work

No matter which person-file you're reading, hold every deliverable to this test
before calling it done: **does it still work with the WAN cable physically
unplugged?** The only feature in the entire system allowed to depend on
connectivity is the "Export to HQ" button, and even that must degrade to "queued
locally, sent when a channel is available" rather than failing outright. If
you're about to add a cloud API call, a SaaS auth provider, or any background
sync — stop; it doesn't belong in this system by design (see Architecture §1,
PRD §9, §11).
