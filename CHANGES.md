# TRINETRA — Approved Changes & Implementation Plan

This document finalizes which proposed enhancements to TRINETRA are approved for implementation, in what order, why each one matters, and exactly what it changes about the system. It supersedes the raw enhancement catalog and reconciles both feasibility reviews into one decision record.

---

## Priority 1 — Current Sprint (Build First)

These are cheap, low-risk, and close gaps that directly undermine the system's core credibility (evidentiary integrity, operational reliability) if left unaddressed.

### 1. Hash-Chained Tamper-Evident Logging
- **What it is:** Every log entry's hash includes the previous entry's hash (`SHA-256(current_entry + previous_hash)`), forming an append-only chain where any retroactive edit is provably detectable.
- **Use / value:** Makes evidence legally defensible. Border incidents can end up in a Court of Inquiry or legal proceeding — being able to mathematically prove no alert or clip metadata was altered after the fact is a decisive credibility feature, and it's what separates TRINETRA from a standard commercial NVR.
- **What it changes in the project:** Adds a hash column and chain-verification logic to the log-write path; adds a "verify integrity" endpoint on the dashboard. Cost is under 1ms of CPU per write — negligible impact on real-time performance.
- **Extends to:** Full evidence log (not just alerts) should use the same mechanism, covering clips and annotations too — build this alongside rather than deferring it.

### 2. Confidence-Propagating Alert Tiers
- **What it is:** Replace binary triggered/not-triggered alerts with a compound probability (`P_threat = P_detection × P_geo-breach × P_behavior`).
- **Use / value:** Prevents a single weak signal (e.g., a person near the fence with a low-confidence weapon flag) from sounding a full alarm. Routes marginal cases to passive logging instead of waking up the whole post — directly reduces alert fatigue.
- **What it changes in the project:** Modifies the alert-severity logic in the rule engine; adds a displayed confidence percentage ("Alert confidence: 73% — moderate, proceed with standard protocol") to the dashboard UI.

### 3. Disk Space Autopurge + RTSP Stream Health Monitoring
- **What it is:** Automated disk-usage monitoring with oldest-first purge above a configurable threshold, plus per-camera stream health tracking (uptime, reconnects, latency, freeze duration).
- **Use / value:** Prevents two silent failure modes that would otherwise go unnoticed at an unattended border post: a filled disk crashing the database, and a frozen/degraded camera feed nobody realizes has stopped working. Also feeds the system's own "per-camera trust score" concept.
- **What it changes in the project:** Adds two lightweight background monitoring workers (cron/async) and two dashboard widgets. No architectural change — pure operational hygiene.

### 4. Shift Handover & Automated SITREP Protocol
- **What it is:** A one-click report summarizing the last 8–12 hours: active alerts, camera uptime, unresolved trajectories, disk/model health.
- **Use / value:** Mirrors existing military shift-handover protocol (jawans already do this manually in a logbook). Reduces knowledge loss between shifts and gives the incoming commander a clear briefing point instead of reconstructing context from scratch.
- **What it changes in the project:** A backend report-generation script querying `entity_log`; exportable as PDF/JSON. No changes to core detection pipeline.

### 5. Disaster Recovery & Backup
- **What it is:** Automated daily snapshots of the VA server database and configuration, with a documented restore procedure.
- **Use / value:** This was originally filed as a "future roadmap" item but is corrected here — it's a baseline requirement, not an enhancement. A single-node deployment at a remote post with no backup strategy is one hardware failure away from losing all evidence and configuration.
- **What it changes in the project:** Adds a scheduled backup job and a tested restore runbook. Should ship with the first production deployment, not be deferred.

### 6. Containerized Model Serving
- **What it is:** Deploy inference models as containerized services (e.g., via Triton) with version control, instead of hard-coded model files.
- **Use / value:** Foundational — enables safe rollback if a model update degrades accuracy, and makes several other approved items (bias auditing, confidence tiering refinement) possible to iterate on safely without redeploying the whole stack.
- **What it changes in the project:** Changes the model-deployment mechanism only; no change to detection logic itself.

**Revision to automatic retraining:** Automatic background fine-tuning is **rejected in its automatic form**. Running backpropagation on the same GPU actively doing real-time inference across 15 RTSP streams causes frame drops and latency spikes during live operation. **Approved instead:** passively accumulate hard negatives automatically, but restrict actual model fine-tuning to manual, commander-scheduled maintenance windows.

---

## Priority 2 — Next Sprint (High Demo/Operational Value)

Feasible without architectural risk, and each directly answers a question evaluators or field commanders are likely to ask.

### 7. BOP Typology Classification (Terrain Presets)
- **What it is:** A configuration registry (`riverine.json`, `fenced_plains.json`, `desert_scrub.json`) that auto-tunes sensitivity, shadow suppression, and tracking decay windows per terrain type.
- **Use / value:** Proves the platform generalizes across sectors (Punjab plains vs. Sundarbans riverine border, for example) rather than being tuned for one demo environment. Directly pre-empts a likely evaluator question.
- **What it changes in the project:** Adds a setup-time typology questionnaire and a template registry; commander retains override capability. No change to core models.

### 8. Multilingual Search Confidence Scoring
- **What it is:** Expose the NER/LLM extraction model's confidence score in the forensic search UI (e.g., green chip for a confidently-extracted time range, yellow for an ambiguous clothing-color match).
- **Use / value:** Removes black-box behavior from the natural-language search feature — commander knows when to trust a result versus fall back to manual filters.
- **What it changes in the project:** UI-layer addition only; surfaces a softmax score the model already produces internally.

### 9. Mobile Commander Companion (Local LAN Only)
- **What it is:** A responsive web client, served locally from the VA server, restricted to the base's own Wi-Fi/mesh network — explicitly **not** internet-facing.
- **Use / value:** Lets the commander check alerts while physically inspecting the perimeter, without leaving the security bubble the sovereignty design depends on.
- **What it changes in the project:** A lightweight frontend reusing the existing backend API; no new data leaves the BOP.

### 10. Hands-Free Voice Tagging (Keyword Spotting)
- **What it is:** A small on-device keyword-spotting model (e.g., a quantized ~50MB model) listening for a fixed, deliberately narrow command set: **"Mark False," "Sound Alarm," "Track Target."**
- **Use / value:** Jawans on duty at odd hours are often in tactical gear or gloves — hands-free interaction with the alert system is a genuinely useful field feature and a strong live-demo moment.
- **What it changes in the project:** Adds a local keyword-spotter service and a command→action mapping in the dashboard backend.
- **Safety correction applied:** The command set deliberately **excludes** any dispatch/action-triggering phrase like "dispatch QRT" from the earlier draft — a misheard command in a high-stress environment must never itself trigger a physical action. Only status-marking and passive commands are voice-triggered; anything consequential still requires manual confirmation.

### 11. Adaptive Compression for Radio/VHF Burst Payloads
- **What it is:** Transcode flagged clips to a bandwidth-efficient codec and package alert metadata into small (~200-byte) serialized payloads for transmission over low-bandwidth tactical VHF links.
- **Use / value:** A full video clip cannot travel across narrow-bandwidth border comms; this makes export actually usable over existing BSF radio infrastructure rather than only over a hypothetical broadband link.
- **What it changes in the project:** Adds a transcoding step to the export pipeline and a compact metadata-only payload format as a fallback when bandwidth is severely constrained.
- **Correction applied:** Use **H.265 only**, not AV1. AV1 encoding is significantly more GPU-intensive than H.265, and would compete for the same GPU resources the inference pipeline needs — exactly during an active incident, which is the worst possible time for that contention. This mirrors the same GPU-contention logic that ruled out automatic retraining (item 6).

### 12. Simple Trajectory Intercept-Point Overlay (Revised Scope)
- **What it is:** A narrower version of the original "QRT deployment optimization" idea — display the existing Kalman-filter trajectory prediction as a predicted intercept point and ETA on the map, without building new floor-plan pathfinding.
- **Use / value:** The original full pathfinding feature was rejected as unnecessary UI clutter for a compact, terrain-familiar outpost — jawans know their base layout. But the system already computes trajectory predictions internally; surfacing that as a simple "predicted point + time" overlay is a near-zero-cost way to make existing work more actionable, without adding the complexity that was correctly rejected.
- **What it changes in the project:** A map-layer overlay only, using data the system already produces. No new pathfinding engine, no new UI subsystem.

---

## Rejected / Deferred — Not Being Built

| Item | Reason |
|---|---|
| **Distributed edge-offload (per-camera AI chips)** | Contradicts the core mandate of using low-cost, standard IP cameras with no edge compute. Exposing AI chips at 15 camera poles to weather and physical/sniper tampering increases cost and attack surface for no benefit at this scale. |
| **Multi-VA server cluster / Postgres multi-master** | Over-engineering for an 8–15 camera outpost; a single workstation GPU handles this load. Distributed database sync at a power-unstable remote post adds failure points without solving a problem TRINETRA currently has. |
| **Fog computing node for clustered BOPs** | Directly violates the sovereignty design principle. BOPs frequently lose *inter-post* connectivity (terrain cuts, line-of-sight, jamming) — a shared regional node isn't just a principle violation, it's often physically unreachable exactly when it would be needed most. |
| **Floor-plan pathfinding for QRT dispatch** | UI clutter for a compact, open-terrain compound jawans already know intimately. (Partially retained — see item 12 above for the scoped-down replacement.) |
| **Full "dispatch QRT" voice command** | Rejected as a voice-triggerable action specifically — a misheard command must never directly cause a physical dispatch. Retained only as a manual, confirmed action. |

---

## Recommended Build Order

| Phase | Items |
|---|---|
| **Sprint 1 (Current)** | Hash-chained logging (1) → Confidence tiers (2) → Disk/RTSP monitoring (3) → Shift handover (4) → Backup/restore (5) → Containerized serving (6) |
| **Sprint 2 (Next)** | BOP typology presets (7) → Search confidence scoring (8) → Mobile companion (9) |
| **Final Polish / Demo** | Voice tagging (10) → Radio/VHF compression, H.265 only (11) → Trajectory intercept overlay (12) |
| **Explicitly not built** | Edge-offload, multi-server cluster, fog computing, floor-plan pathfinding, voice-triggered dispatch |

**Sequencing note:** Item 6 (Containerized model serving) should land early in Sprint 1 even though it has no standalone demo value, because it's what makes safe iteration on the confidence-tiering and future bias-auditing work possible without redeploying the whole stack.

---

## One Open Item Needing Validation Before Sprint 1 Closes

The claim that a single workstation GPU with TensorRT "easily" handles concurrent Re-ID + FRS + ANPR + tracking across 15 RTSP streams is currently asserted without a benchmark. Given this exact number is used to justify rejecting the multi-server cluster item, it should be benchmarked and documented before that rejection is presented as final — an unsupported performance claim is the most likely place an evaluator would push back.

---

*This document reflects the consolidated outcome of two independent feasibility reviews of the TRINETRA enhancement catalog, with corrections applied where the second review identified a stronger technical argument than the first.*
