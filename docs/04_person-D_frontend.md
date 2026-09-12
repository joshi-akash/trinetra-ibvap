# Person D — Frontend & Integration Lead

**You own:** `frontend/` in the shared monorepo, plus integration testing, the
demo script, and the BOP setup runbook (jointly with Person C).
**You build:** the entire React commander dashboard — every screen the SRS/UI-UX
doc specifies — and you are the person who spends the most time exercising the
*whole* system end-to-end, so you're also the team's early-warning system for
integration breaks.
**Read first:** `docs/00_INTEGRATION_CONTRACT.md` — the REST/WebSocket contract
in its §2 is the exact surface you build against, from Day 1, using mocked
responses before Person C's real backend exists.

Hand this file, with the integration contract, to an AI coding agent to build
from scratch.

---

## 1. What You're Building, in One Paragraph

Every other person's work is invisible until it shows up on this dashboard: the
live camera grid, the alert feed with siren and flash on a real breach, the
geo-fence editor a commander draws polygons on, the forensic search bar, the
export button, the false-flag control. The system's one hard requirement —
**it must work with zero connectivity to Delhi HQ** — is something *you* prove
every time you load this dashboard with the WAN cable pulled, because if the
frontend silently breaks offline, the sovereignty requirement is dead no matter
how correct the backend is.

---

## 2. Tech Stack (do not substitute without team sign-off)

| Component | Choice | Why |
|---|---|---|
| Framework | React + TypeScript | TypeScript reduces schema-drift bugs against Person C's API — a mismatched field name fails at compile time, not in front of an evaluator |
| Styling | Tailwind CSS | Component reuse for the live-feed grid, map panel, alert feed |
| Real-time | WebSocket client against `/ws/alerts` | Live alert push, live bounding-box overlays, trust-score updates — avoid polling |
| Serving | Served directly from the VA server (nginx container) over the BOP LAN | No external hosting, no CDN dependency — this is itself part of the sovereignty requirement |

Install: `npm create vite@latest frontend -- --template react-ts`, then add
`tailwindcss`, `react-router-dom`, and a WebSocket client library of your choice.

---

## 3. Screens You Own (from the UI/UX Document)

Build every one of these for MVP — cross-reference the UI/UX Document's
Section 7 (Screen Specifications) for exact layout, states, and copy:

| Screen | Core behavior |
|---|---|
| Live View (default landing) | Real-time-style camera grid with bounding-box overlays |
| Alert Feed | Card layout, thumbnail, confidence tag, "View clip"/"Export" actions; siren + flash on new active alert |
| Alert Detail | Full package view — JSON fields, thumbnail, 30s clip, which rule fired and at what confidence |
| Geo-Fence Editor | Draw/edit polygon on a live feed, minimum 3-vertex validation |
| Forensic Search | Filter-picker UI (time range, camera, entity type, color) — structured filters only for MVP, no free-text NL search |
| Camera Management | List cameras, trust/status, calibration entry point |
| Review Queue | False-flag marking, reflects into `false_flag_log` |
| Export / HQ Log | Select a record, trigger export, see export history |
| Settings | Role-appropriate config |
| Digital Twin Map (P2/stretch) | Camera FOV cones + live entity dots on a 2D map — build only if Weeks 1–3 core screens are solid ahead of schedule |

Every screen needs explicit loading, empty, and error states per UI/UX
Document §11 — this is not optional polish, it's in your Week 5 functional-pass
checklist below.

---

## 4. Folder Layout (inside `frontend/`)

```
frontend/
└── src/
    ├── api/
    │   └── client.ts              # typed client generated/hand-written against docs/api-contract.yaml
    ├── ws/
    │   └── alertsSocket.ts        # /ws/alerts connection + reconnect handling
    ├── screens/
    │   ├── LiveView/
    │   ├── AlertFeed/
    │   ├── AlertDetail/
    │   ├── GeoFenceEditor/
    │   ├── ForensicSearch/
    │   ├── CameraManagement/
    │   ├── ReviewQueue/
    │   ├── ExportLog/
    │   ├── Settings/
    │   └── DigitalTwinMap/         # stretch
    ├── components/                 # Camera Tile, Alert Card, Filter Chip, Trust Score Badge, Geo-Fence Polygon Tool, Timeline Scrubber, Status Pill — from UI/UX Document §9
    └── mocks/
        └── mockApi.ts              # mocked responses matching the frozen contract exactly — your Week 1–2 lifeline
```

---

## 5. Week-by-Week Build Order

### Week 1 — Scaffold + Contract Review
- **Mon:** Scaffold the React+TS+Tailwind project. Review the UI/UX Document; list every screen/state needed for MVP (the table in §3 above is your starting checklist).
- **Tue:** Build static component shells for the Live View grid + Alert Feed (no data yet).
- **Wed:** Build Camera Registry + Geo-Fence Editor shells.
- **Thu:** Review Person C's draft OpenAPI spec — flag anything the UI needs that isn't there. This is your one chance to shape the contract before it freezes; use it.
- **Fri:** Build the API client layer against the **frozen** contract, pointed at mocked responses in `mocks/mockApi.ts`.

**Friday checkpoint:** API client exists and compiles against the frozen contract; mocked responses match the contract's shapes exactly (this matters — mocks that drift from the real contract are how integration debt hides until Week 4).

### Week 2 — Full Navigation on Mocks
- **Mon:** Live View: real-time-style feed grid (mocked frames + boxes).
- **Tue:** Alert Feed: card layout, thumbnail, confidence tag, "View clip"/"Export" buttons (mocked).
- **Wed:** Geo-Fence Editor: draw/edit polygon on a live (mocked) feed, ≥3-vertex validation.
- **Thu:** Forensic Search: filter-picker UI (time, camera, entity type, color) — Layer 1 only.
- **Fri:** Wire mocked data through the full navigation flow; catch missing states (empty/loading/error) against the UI/UX doc.

**Friday checkpoint:** all MVP screens exist and are navigable on mock data.

### Week 3 — Switch to Real Data
- **Mon:** Point Live View at the real `/api/entities` stream.
- **Tue:** Point Alert Feed at the real `/api/alerts`; verify siren/flash actually trigger on a real alert from Person C's backend.
- **Wed:** Wire Forensic Search to the real search endpoint.
- **Thu:** Wire the Export button to the real endpoint; wire the false-flag "mark" action end-to-end.
- **Fri:** Build the basic 2D Digital Twin map UI, consuming real entity locations (coordinate with Person C, who supplies the data — this is P2/stretch, only if on schedule).

**Friday checkpoint:** dashboard is live-data-driven for Live View, Alert Feed, Search, Export — not mocks.

### Week 4 — Integration (joint week, see integration contract §5)
Same joint schedule as everyone else: remove mocks Monday, fix-and-rerun Tuesday,
WAN-disconnected re-run Wednesday (this is the day your work matters most —
confirm the dashboard loads and remains fully functional with the WAN cable
physically pulled, per SRS/PRD acceptance criteria), scripted false-flag and
multi-modal-corroboration scenarios Thursday, M4 sign-off Friday.

### Week 5 — Hardening
- **Mon:** Functional pass against every screen's loading/error/empty states (UI/UX doc checklist) — go screen by screen, don't sample.
- **Tue:** Accessibility pass — contrast, keyboard navigation (UI/UX doc §13).
- **Wed–Thu:** Bug triage and fix cycle with the team.
- **Fri:** Regression pass on everything touched this week.

### Week 6 — Deployment & Demo Readiness
- **Mon–Tue:** Co-write the BOP setup runbook with Person C — camera config, geo-fence seeding, calibration steps, in plain literal steps with no assumed context.
- **Wed:** Participate in the non-author standup test — watch where someone unfamiliar with the system gets confused by the UI specifically, fix it.
- **Thu:** Write the demo script matching anticipated evaluator questions (SRS §18 / PRD §7 user stories). Run dry-run #1, including one WAN-disconnected segment.
- **Fri:** Dry-run #2. Help test the fallback plan (recorded footage / backup camera) for real, not just on paper.

---

## 6. Coordination Points

- **Person C:** your only real interface — everything you build talks to their
  API/WebSocket contract. Any UI need that isn't covered by the current
  contract gets raised immediately, not worked around with a one-off mock that
  never becomes real.
- **Person A/B:** no direct interface. If something in the alert feed shows the
  wrong alert type, wrong confidence, or a field that should never be null is
  null, trace it backward through Person C's API response to whichever AI
  stage actually produced that value before assuming it's a frontend rendering
  bug.

---

## 7. Definition of Done for Your Stage

1. Every screen tested with WAN physically disconnected — the only feature
   allowed to visibly depend on connectivity is the "Export to HQ" transport
   step itself (the package build and queueing must still work offline).
2. Every screen traces to a specific UI/UX Document section and SRS/PRD
   requirement.
3. Has a scripted test with a stated expected outcome (not "looks right").
4. Every active-alert display shows which rule fired and at what confidence
   (this is a real SRS acceptance criterion, not a nice-to-have).
5. Loading, empty, and error states exist for every screen — not just the
   happy path.
6. A second person looked at it before it's marked done.

---

## 8. Prompt You Can Hand an AI Coding Agent

> Build the `frontend/` portion of the TRINETRA monorepo described in
> `docs/00_INTEGRATION_CONTRACT.md`, following the screen specifications in the
> project's UI/UX Document. Use React + TypeScript + Tailwind CSS. Implement
> Live View (camera grid with bounding-box overlays), Alert Feed (with siren/
> flash on new active alerts via a `/ws/alerts` WebSocket connection), Alert
> Detail, Geo-Fence Editor (polygon drawing on a live feed, ≥3-vertex
> validation), Forensic Search (structured filters only — time range, camera,
> entity type, color), Camera Management, Review Queue (false-flag marking),
> Export/HQ Log, and Settings. Build a typed API client against
> `docs/api-contract.yaml` exactly, and a matching mock layer for development
> before the real backend exists. Every screen needs explicit loading, empty,
> and error states. The entire dashboard must load and remain fully functional
> with no WAN connectivity — the only network-dependent action anywhere in the
> UI is the literal HQ export transport step.
