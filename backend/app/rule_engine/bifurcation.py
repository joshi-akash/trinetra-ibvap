from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from backend.app.schemas import FrameAnalysis, EntityDetection
from backend.app.rule_engine.geofence_check import geofence_evaluator
from backend.app.rule_engine.condition_wiring import condition_wiring

@dataclass
class EvaluationDecision:
    entity_detection: EntityDetection
    is_alert: bool
    alert_type: Optional[str]  # geo_fence | behavior | tamper | correlated
    rule_fired: Optional[str]
    confidence_score: float
    retention_tier: str  # passive | protected
    signals_detected: List[str]


class BifurcationEngine:
    """
    Decides passive-log vs active-alert (FR-ALR-01, FR-ALR-02, FR-ALR-04).
    This is the single most important piece of backend logic in TRINETRA.
    """

    def evaluate_frame(
        self,
        frame_analysis: FrameAnalysis,
        geofence_coords: Optional[List[List[float]]] = None
    ) -> Tuple[List[EvaluationDecision], Optional[Dict[str, Any]]]:
        """
        Evaluates a frame analysis object.
        Returns:
            (List of entity decisions, Optional camera-level tamper alert)
        """
        decisions: List[EvaluationDecision] = []
        camera_alert = None

        # 1. Camera Tamper Check (Darkness-aware)
        is_low_light = bool(frame_analysis.frame_quality.low_light)
        tamper_signal = frame_analysis.frame_quality.tamper_signal.model_dump()
        is_tampered, tamper_reason = condition_wiring.is_camera_tampered(tamper_signal, is_low_light=is_low_light)
        if is_tampered:
            camera_alert = {
                "camera_id": frame_analysis.camera_id,
                "timestamp": frame_analysis.timestamp,
                "alert_type": "tamper",
                "rule_fired": f"camera_tamper_detected: {tamper_reason}",
                "confidence_score": 0.95,
                "retention_tier": "protected"
            }

        # 2. Evaluate Each Detected Entity
        for entity in frame_analysis.entities:
            signals = []

            # Condition A: Geo-Fence Intersection (FR-ALR-02.1)
            is_breach = geofence_evaluator.is_inside_geofence(
                entity_lat=entity.location.lat,
                entity_lon=entity.location.lon,
                geofence_coords=geofence_coords
            )
            if is_breach:
                signals.append("geo_fence_breach")

            # Condition B: Behavioral Posture & Props (FR-ALR-02.2) with night adaptive floor
            posture = entity.attributes.posture
            if condition_wiring.is_suspicious_posture(posture, confidence=entity.confidence, is_low_light=is_low_light):
                signals.append(f"suspicious_posture:{posture}")

            threat_props = condition_wiring.get_threat_props_found(entity.attributes.props, confidence=entity.confidence, is_low_light=is_low_light)
            for prop in threat_props:
                signals.append(f"threat_prop:{prop}")

            # Condition C: Known Suspect Face Match
            face_dict = entity.attributes.face_match.model_dump() if entity.attributes.face_match else None
            if condition_wiring.is_suspect_face_match(face_dict):
                signals.append(f"suspect_face_match:{face_dict.get('suspect_id')}")

            # Condition D: Night Stealth Aggravation (Darkness + Breach + Crouching/Crawling)
            has_posture = any("suspicious_posture" in s for s in signals)
            if is_low_light and is_breach and has_posture and condition_wiring.is_night_stealth_escalation_enabled():
                signals.append("night_stealth_approach")

            # 3. Decision Bifurcation Logic (FR-ALR-01 vs FR-ALR-02 vs FR-ALR-04)
            if not signals:
                # FR-ALR-01: Passive Logging (Default Path)
                decision = EvaluationDecision(
                    entity_detection=entity,
                    is_alert=False,
                    alert_type=None,
                    rule_fired=None,
                    confidence_score=entity.confidence,
                    retention_tier="passive",
                    signals_detected=[]
                )
            else:
                # FR-ALR-04: Multi-Modal Corroboration for Highest Severity Tier
                has_weapon = any("threat_prop:weapon" in s for s in signals)
                has_night_stealth = "night_stealth_approach" in signals

                if (has_weapon and (is_breach or has_posture)) or has_night_stealth or len(signals) >= 2:
                    alert_type = "correlated"
                    if has_night_stealth:
                        rule_fired = "night_stealth_intrusion: " + " & ".join(signals)
                    else:
                        rule_fired = "multi_modal_corroboration: " + " & ".join(signals)
                    confidence = min(1.0, entity.confidence + 0.1)
                elif is_breach:
                    alert_type = "geo_fence"
                    rule_fired = "geofence_polygon_breach"
                    confidence = entity.confidence
                else:
                    alert_type = "behavior"
                    rule_fired = "behavioral_anomaly: " + " & ".join(signals)
                    confidence = entity.confidence

                decision = EvaluationDecision(
                    entity_detection=entity,
                    is_alert=True,
                    alert_type=alert_type,
                    rule_fired=rule_fired,
                    confidence_score=confidence,
                    retention_tier="protected",  # Never auto-purged (FR-RET-02)
                    signals_detected=signals
                )

            decisions.append(decision)

        return decisions, camera_alert

bifurcation_engine = BifurcationEngine()
