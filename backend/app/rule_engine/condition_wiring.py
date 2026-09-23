import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from backend.app.config import settings

class ConditionWiring:
    """Loads and wires dynamic threshold definitions from ai_behavior/thresholds.yaml."""

    def __init__(self, thresholds_path: Optional[Path] = None):
        self.thresholds_path = thresholds_path or settings.THRESHOLDS_FILE
        self.config: Dict[str, Any] = self._load_thresholds()

    def _load_thresholds(self) -> Dict[str, Any]:
        default_config = {
            "geofence": {"enabled": True, "min_confidence": 0.50},
            "behavior": {
                "suspicious_postures": ["crouching", "sprinting", "prone", "crawling", "climbing"],
                "posture_confidence_floor": 0.20,
                "threat_props": [
                    "weapon", "knife", "gun", "rifle", "firearm",
                    "covered_face", "large_backpack", "backpack", "suitcase", "handbag", "big_bag"
                ],
                "prop_confidence_floor": 0.20
            },
            "face_recognition": {"match_confidence_threshold": 0.80},
            "tamper": {
                "min_laplacian_variance": 100.0,
                "variance_drop_percentage": 50.0,
                "enable_drift_detection": True
            },
            "corroboration": {
                "require_two_signals_for_highest_tier": True,
                "highest_tier_threats": ["armed_intruder", "perimeter_breach_tampered", "night_stealth_intrusion"]
            },
            "low_light": {
                "enabled": True,
                "min_lux_threshold": 45.0,
                "adaptive_confidence_discount": 0.10,
                "night_retention_days": 60,
                "night_tamper_variance_floor": 30.0,
                "night_stealth_escalation": True
            },
            "retention": {
                "passive_retention_days": 30,
                "protected_retention_days": -1
            }
        }

        if self.thresholds_path and self.thresholds_path.exists():
            try:
                with open(self.thresholds_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        # Merge loaded over defaults
                        for k, v in loaded.items():
                            if isinstance(v, dict) and k in default_config:
                                default_config[k].update(v)
                            else:
                                default_config[k] = v
            except Exception:
                pass

        return default_config

    def reload(self):
        self.config = self._load_thresholds()

    def is_suspicious_posture(self, posture: Optional[str], confidence: float = 1.0, is_low_light: bool = False) -> bool:
        if not posture:
            return False
        floor = self.get_effective_posture_floor(is_low_light)
        if confidence < floor:
            return False
        suspicious = self.config.get("behavior", {}).get("suspicious_postures", [])
        return posture.lower() in [s.lower() for s in suspicious]

    def get_threat_props_found(self, props: Optional[List[str]], confidence: float = 1.0, is_low_light: bool = False) -> List[str]:
        if not props:
            return []
        floor = self.get_effective_prop_floor(is_low_light)
        if confidence < floor:
            return []
        threat_props = self.config.get("behavior", {}).get("threat_props", [])
        threat_lower = [t.lower() for t in threat_props]
        return [p for p in props if p.lower() in threat_lower]

    def is_suspect_face_match(self, face_match: Optional[dict]) -> bool:
        if not face_match:
            return False
        threshold = self.config.get("face_recognition", {}).get("match_confidence_threshold", 0.80)
        return face_match.get("confidence", 0.0) >= threshold

    def is_camera_tampered(self, tamper_signal: dict, is_low_light: bool = False) -> Tuple[bool, str]:
        if tamper_signal.get("is_tampered", False):
            return True, "tamper_flag_set"
        
        # When in darkness / low-light, adjust blur floor to prevent false alarms from natural low-light noise
        if is_low_light:
            min_var = self.config.get("low_light", {}).get("night_tamper_variance_floor", 30.0)
        else:
            min_var = self.config.get("tamper", {}).get("min_laplacian_variance", 100.0)

        variance = tamper_signal.get("laplacian_variance", 500.0)
        if variance < min_var:
            return True, f"low_laplacian_variance ({variance:.1f} < {min_var})"
        
        if tamper_signal.get("reference_point_drift", False):
            return True, "camera_reference_point_drift"

        return False, ""

    def get_effective_posture_floor(self, is_low_light: bool = False) -> float:
        base_floor = self.config.get("behavior", {}).get("posture_confidence_floor", 0.65)
        if is_low_light and self.config.get("low_light", {}).get("enabled", True):
            discount = self.config.get("low_light", {}).get("adaptive_confidence_discount", 0.10)
            return max(0.40, base_floor - discount)
        return base_floor

    def get_effective_prop_floor(self, is_low_light: bool = False) -> float:
        base_floor = self.config.get("behavior", {}).get("prop_confidence_floor", 0.70)
        if is_low_light and self.config.get("low_light", {}).get("enabled", True):
            discount = self.config.get("low_light", {}).get("adaptive_confidence_discount", 0.10)
            return max(0.40, base_floor - discount)
        return base_floor

    def is_night_stealth_escalation_enabled(self) -> bool:
        return self.config.get("low_light", {}).get("night_stealth_escalation", True)

condition_wiring = ConditionWiring()

