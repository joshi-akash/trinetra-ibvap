# Person C — Backend & Infrastructure Lead

**You own:** `infra/`, `backend/`, and `docs/api-contract.yaml` in the shared
monorepo.
**You build:** Docker/deployment, the database, the FastAPI backend, auth, MQTT,
RTSP ingestion, the Rule Engine's plumbing, the Alert Manager, the Export
Service, and the forensic search backend.
**Read first:** `docs/00_INTEGRATION_CONTRACT.md` — you consume the
`FrameAnalysis` object Person A/B produce (§2), and you are the primary author
of the REST/WebSocket contract Person D builds against (§2, §5). You are the
plumbing everyone else's water runs through — nothing in this system works
without your foundation being solid and early.

Hand this file, with the integration contract, to an AI coding agent to build
from scratch.

---

## 1. What You're Building, in One Paragraph

You are the system's spine: the database that stores every detection, the API
that lets the dashboard read and act on that data, the ingestion pipeline that
gets camera frames to the AI stack in the first place, the rule engine that
decides passive-log-vs-loud-alert, and the export mechanism that makes HQ
reporting a deliberate human action rather than automatic sync. You also own
the whole deployment story — Docker Compose, the golden image, the BOP setup
runbook — because "it works on my dev machine" is not the deliverable; "a
non-author can stand this up on fresh hardware" is.

---

## 2. Tech Stack (do not substitute without team sign-off)

| Component | Choice | Why |
|---|---|---|
| API framework | Python, FastAPI | Async I/O for concurrent camera streams + WebSocket push; native Pydantic validation matches the structured JSON schema requirements |
| Database | PostgreSQL 15+ with PostGIS | Native geography/geometry types for geo-fences and entity locations; GiST spatial indexing |
| Event transport | MQTT (Mosquitto broker) | Lightweight pub/sub between AI pipeline, backend, and dashboard; fully LAN, no internet |
| Video ingestion | GStreamer or FFmpeg with RTSP input | Camera-agnostic, no vendor SDK lock-in |
| Geospatial | Shapely + GEOS | Polygon intersection for geo-fencing |
| Auth | Local JWT (FastAPI), bcrypt/argon2 password hashing | Zero cloud dependency — must work fully offline |
| Containerization | Docker + Docker Compose | Reproducible deployment; each service its own container |
| Process supervision | systemd wrapping the Compose stack | Auto-restart without Kubernetes overhead — this is a single-node-pair deployment, not a cluster |
| Search (MVP) | Structured filter-based search; synonym dictionary layer | NL search is explicitly deferred post-MVP per PRD §6.2 — build the filter/synonym layer only |

Install: `pip install fastapi uvicorn sqlalchemy psycopg2-binary shapely paho-mqtt python-jose[cryptography] passlib[bcrypt] gstreamer-python` (or use `ffmpeg-python` if preferring FFmpeg over GStreamer bindings).

---

## 3. Folder Layout (inside `backend/` and `infra/`)

```
infra/
├── docker-compose.yml            # postgres, mosquitto, backend, frontend (nginx), all services
├── postgres/
│   └── init.sql                  # schema from integration contract §4
├── mosquitto/
│   └── mosquitto.conf
└── systemd/
    └── trinetra.service           # wraps `docker compose up` for boot-time auto-restart

backend/
└── app/
    ├── main.py                    # FastAPI app entrypoint
    ├── api/
    │   ├── auth.py                 # /api/auth/login
    │   ├── cameras.py              # /api/cameras, /api/cameras/{id}/calibrate
    │   ├── geofence.py             # /api/geofence/{camera_id}
    │   ├── entities.py             # /api/entities, /api/entities/{id}/false-flag, /export
    │   ├── search.py               # /api/search
    │   ├── alerts.py               # /api/alerts/active, /api/alerts/{id}/acknowledge
    │   └── ws_alerts.py             # /ws/alerts — MQTT-to-WebSocket bridge
    ├── ingestion/
    │   └── rtsp_service.py         # opens RTSP connections, demuxes frames, ~60s rolling ring buffer per camera
    ├── rule_engine/
    │   ├── geofence_check.py       # Shapely polygon intersection
    │   ├── bifurcation.py          # passive-log vs active-alert decision (the single most important piece of backend logic — get it right early)
    │   └── condition_wiring.py     # reads ai_behavior/thresholds.yaml, wires posture/tamper signals into bifurcation
    ├── alert_manager/
    │   └── package_builder.py      # JSON + WebP thumbnail + 30s clip from ring buffer, MQTT publish
    ├── export_service/
    │   └── export_builder.py       # signed zip/tarball, hands off to whatever transport is available
    ├── search_service/
    │   ├── synonym_dictionary.py   # Layer-1 deterministic term resolution
    │   └── query_translator.py     # resolved filters → parameterized PostGIS query
    └── auth/
        └── jwt_auth.py             # local JWT issue/validate, bcrypt hashing, role-based dependency injection
```

---

## 4. What You Consume From Others

- **From Person A/B:** the `FrameAnalysis` object, called in-process from your
  `ingestion/rtsp_service.py` loop (or via local gRPC if the AI pipeline runs
  as a separate process for GPU isolation — your call, document whichever you
  pick in `docs/00_INTEGRATION_CONTRACT.md` if it changes from the default
  assumption). You write its contents into `entity_log` rows.
- **From Person B specifically:** `ai_behavior/thresholds.yaml` — read this
  rather than hardcoding trigger thresholds, so tuning during Week 3/5 doesn't
  require a backend redeploy.

## 5. What Others Consume From You

- **Person D** builds the entire frontend against your `docs/api-contract.yaml`
  (OpenAPI) plus the `/ws/alerts` WebSocket. Draft this by Thursday of Week 1;
  freeze it Friday with all 4 signing off. After freeze, any change needs
  same-day 4-way sign-off — see integration contract §2, §5.
- **Everyone** relies on your `docker-compose.yml` to actually run the system
  locally — keep it current from Week 1 onward, not something assembled at the
  end.

---

## 6. Week-by-Week Build Order

### Week 1 — Environment + Foundation
- **Mon:** Provision/verify the GPU box, install CUDA/cuDNN/Docker. Set up the repo skeleton (`backend/`, `frontend/`, `ai_detection/`, `ai_behavior/`, `infra/`).
- **Tue:** Stand up the PostgreSQL+PostGIS container. Write `entity_log`, `camera_registry`, `false_flag_log` migrations from the schema in the integration contract §4.
- **Wed:** Stand up the Mosquitto MQTT container. Write the FastAPI skeleton with stub endpoints for every route in §3.
- **Thu:** **Draft the OpenAPI spec** covering `/entities`, `/alerts`, `/cameras`, `/search`, `/export`, `/auth`. This is the day's critical deliverable — Person D needs it to start real UI work Friday.
- **Fri:** **Freeze the API contract** (all 4 sign off, commit the YAML — not left in chat/notes). Implement JWT auth (single role for MVP simplicity, extend to full role matrix as time allows).

**Friday checkpoint:** `docker compose up` gives an empty-but-running stack; API contract frozen in writing.

### Week 2 — Real Data Path
- **Mon:** Implement the RTSP ingestion service with the rolling ~60s ring buffer.
- **Tue:** Implement the `/api/entities` write path against the real schema, receiving `FrameAnalysis` objects from Person A/B's pipeline.
- **Wed:** Implement the Shapely geo-fence polygon check — the first real Rule Engine condition.
- **Thu:** Implement the passive/active bifurcation logic (FR-ALR-01/02). Get this right early — it's the single most consequential piece of logic in the backend; every alert in the system routes through it.
- **Fri:** Implement alert package generation (JSON + WebP thumbnail + 30s clip pulled from the ring buffer, FR-ALR-03).

**Friday checkpoint:** a person walking into a mocked/real geo-fence produces a stored `entity_log` row with `is_alert = true`, a clip file, and a thumbnail — even if not yet visible in the real UI.

### Week 3 — Search, Export, Behavior Wiring
- **Mon:** Wire Person B's behavior triggers into the Rule Engine as a second active-alert condition.
- **Tue:** Implement the Layer-1 synonym/canonical dictionary search backend (FR-SRCH-01).
- **Wed:** Implement `/api/search` execution against the real schema (structured filters + synonym resolution).
- **Thu:** Implement the export packaging endpoint (JSON+thumbnail to local file, FR-EXP-01/02).
- **Fri:** Basic 2D Digital Twin map data (camera FOV cones + live entity locations) — coordinate closely with Person D, who builds the actual map UI. This is P2/stretch — don't let it crowd out anything above.

**Friday checkpoint:** every AI capability has produced at least one real, verifiable log row through the real pipeline; Search and Export endpoints work against real data.

### Week 4 — Integration (joint week, see integration contract §5)
- **Mon:** Remove all remaining mocks; run the full loop once, end-to-end. Log every break.
- **Tue:** Fix breaks. Repeat-until-green.
- **Wed:** **Physically disconnect WAN.** Re-run the entire loop with zero network access. Confirm export still produces a local file — it must never depend on a live transport.
- **Thu:** Run the false-flag path (mark → `false_flag_log` write → UI reflects it) and the multi-modal corroboration path (two signals required for the highest severity tier) as separate scripted scenarios.
- **Fri:** **M4 sign-off** — confirm all four points in the integration contract §5 with the whole team, live.

### Week 5 — Hardening
- **Mon:** Load test — sustained multi-camera ingestion, watch for ring-buffer memory growth over hours.
- **Tue:** Security pass — confirm auth can't be bypassed, export/deletion actions are audit-logged (`audit_log` write on every manual deletion, false-flag mark, export, admin change).
- **Wed:** Bug triage jointly with the team, severity-ranked (crashes/data-loss first).
- **Thu:** Fix cycle, working the ranked list top-down.
- **Fri:** Regression pass on everything touched this week; start the multi-hour unattended soak run.

### Week 6 — Deployment & Demo Readiness
- **Mon:** Check soak-run results, fix anything that surfaced. Build the golden Docker image / deployment bundle.
- **Tue:** Write the BOP setup runbook (camera config, geo-fence seeding, calibration steps) in plain, literal steps — no assumed context. Joint with Person D.
- **Wed:** Support the non-author standup test — someone who didn't build the system follows the runbook on fresh hardware, zero help. Fix every point of confusion, especially in your deployment steps.
- **Thu–Fri:** Support demo script + both dry-runs, particularly the WAN-disconnected segment and the tested (not just documented) fallback plan.

---

## 7. Definition of Done for Your Stage

1. Tested with WAN physically disconnected (except the export mechanism's actual transport step, which is inherently connectivity-dependent — but the export *package build* must succeed offline).
2. Every endpoint traces to a specific SRS/PRD line item.
3. Has a scripted test with a stated expected outcome.
4. Every alert path in the Rule Engine logs which rule fired and at what confidence.
5. No path exists where a `retention_tier = protected` record can be silently deleted — deletion always requires an explicit admin action and always writes `deletion_audit`.
6. A second person looked at it before it's marked done.

---

## 8. Prompt You Can Hand an AI Coding Agent

> Build the `backend/` and `infra/` portions of the TRINETRA monorepo described
> in `docs/00_INTEGRATION_CONTRACT.md`. Implement a FastAPI backend with local
> JWT authentication (bcrypt-hashed passwords, role-based access for
> admin/commander/operator), a PostgreSQL+PostGIS database using the exact
> schema in the integration contract §4, an RTSP ingestion service with a
> rolling ~60-second ring buffer per camera, a Rule Engine that evaluates
> Shapely geo-fence intersection and behavioral/tamper triggers (read from
> `ai_behavior/thresholds.yaml`) to decide passive-log vs active-alert, an
> Alert Manager that assembles JSON+thumbnail+30s-clip packages and publishes
> over MQTT (Mosquitto), a manual (never automatic) HQ export service producing
> a self-contained signed package, and a structured/synonym-based forensic
> search backend. Every endpoint must match `docs/api-contract.yaml` exactly.
> The entire stack must run via `docker compose up` and remain fully functional
> with no WAN connectivity except the literal HQ export transport step. Write
> migrations, a docker-compose.yml wiring Postgres/Mosquitto/backend/frontend
> together, and scripted tests for the geo-fence breach path, the false-flag
> path, and the export path.
