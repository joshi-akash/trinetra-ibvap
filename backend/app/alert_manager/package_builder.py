import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models import EntityLog
from backend.app.ingestion.rtsp_service import rtsp_service

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Assembles alert packages (JSON + WebP thumbnail + 30s video clip, FR-ALR-03)
    and dispatches them over MQTT and WebSocket.
    """

    def __init__(self):
        self._mqtt_client = None
        self._init_mqtt()

    def _init_mqtt(self):
        try:
            import paho.mqtt.client as mqtt
            if hasattr(mqtt, "CallbackAPIVersion"):
                self._mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            else:
                self._mqtt_client = mqtt.Client()
            self._mqtt_client.connect_async(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, 60)
            self._mqtt_client.loop_start()
            logger.info("Connected to Mosquitto MQTT broker")
        except Exception as e:
            logger.warning(f"MQTT broker not reachable or paho-mqtt not installed ({e}). Running in offline bridge mode.")
            self._mqtt_client = None

    def publish_mqtt_event(self, topic: str, payload: dict):
        if self._mqtt_client:
            try:
                self._mqtt_client.publish(topic, json.dumps(payload, default=str), qos=1)
            except Exception as e:
                logger.error(f"Failed to publish MQTT message: {e}")

    def build_and_save_alert(
        self,
        db: Session,
        camera_id: str,
        timestamp: datetime,
        entity_detection: Any,
        alert_type: str,
        rule_fired: str,
        confidence_score: float,
        frame_ref: Optional[str] = None,
        is_low_light: bool = False,
        posture: Optional[str] = None
    ) -> EntityLog:
        """
        Builds the complete FR-ALR-03 Alert Package:
        1. Unique alert UUID
        2. 30-second evidence clip from ring buffer
        3. WebP compressed thumbnail (with night CLAHE enhancement if low light)
        4. Stored in entity_log with retention_tier='protected'
        5. Dispatched to MQTT topic trinetra/alerts & WebSocket bridge
        """
        alert_id = str(uuid.uuid4())
        
        # 1. Extract 30s Evidence Clip & WebP Thumbnail from Ring Buffer
        clip_path = rtsp_service.extract_30s_clip(
            camera_id=camera_id,
            trigger_time=timestamp,
            alert_id=alert_id
        )
        
        thumbnail_path = rtsp_service.extract_thumbnail(
            camera_id=camera_id,
            trigger_time=timestamp,
            alert_id=alert_id,
            bbox=entity_detection.bbox if hasattr(entity_detection, 'bbox') else None,
            is_low_light=is_low_light
        )

        # 2. Database Record Persistence
        attrs = entity_detection.attributes if hasattr(entity_detection, 'attributes') else None
        loc = entity_detection.location if hasattr(entity_detection, 'location') else None
        detected_posture = posture or (attrs.posture if attrs else None)

        entity_log = EntityLog(
            id=alert_id,
            camera_id=camera_id,
            timestamp=timestamp,
            entity_type=entity_detection.entity_type if hasattr(entity_detection, 'entity_type') else "unknown",
            upper_color=attrs.upper_color if attrs else None,
            lower_color=attrs.lower_color if attrs else None,
            height_cm=attrs.height_cm if attrs else None,
            gender=attrs.gender if attrs else None,
            plate_text=attrs.plate_text if attrs else None,
            vehicle_type=getattr(attrs, "vehicle_type", None) if attrs else None,
            direction=getattr(attrs, "direction", None) if attrs else None,
            speed_kmh=getattr(attrs, "speed_kmh", None) if attrs else None,
            face_name=getattr(attrs, "face_name", None) if attrs else ((attrs.face_match.suspect_id if attrs.face_match else None) if attrs else None),
            skin_tone=getattr(attrs, "skin_tone", None) if attrs else None,
            location_lat=loc.lat if loc else None,
            location_lon=loc.lon if loc else None,
            trajectory_id=entity_detection.track_id if hasattr(entity_detection, 'track_id') else None,
            is_alert=True,
            alert_type=alert_type,
            confidence_score=confidence_score,
            clip_path=clip_path,
            thumbnail_path=thumbnail_path,
            retention_tier="protected",  # Never auto-purged (FR-RET-02)
            deleted_manually=False,
            rule_fired=rule_fired,
            acknowledged=False,
            is_low_light=is_low_light,
            posture=detected_posture
        )

        # 2b. Compute Clip Digest & Cryptographic Notarization Token
        clip_sha = "0" * 64
        try:
            if clip_path and Path(clip_path).exists():
                with open(clip_path, "rb") as f:
                    clip_sha = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass

        entity_log.clip_sha256 = clip_sha
        from backend.app.crypto_ledger.merkle_ledger import generate_notarization_token, merkle_ledger
        token = generate_notarization_token(alert_id, clip_sha, timestamp.isoformat(), camera_id)
        entity_log.notarization_token = token

        db.add(entity_log)
        db.commit()
        db.refresh(entity_log)

        # 3. Assemble Alert Payload for Real-Time Dispatch (Conforming to Contract 2)
        alert_payload = {
            "event_type": "ACTIVE_ALERT",
            "type": "NEW_ALERT",
            "alert_id": alert_id,
            "camera_id": camera_id,
            "timestamp": timestamp.isoformat(),
            "alert_type": alert_type,
            "rule_fired": rule_fired,
            "confidence_score": confidence_score,
            "threat_score": confidence_score,
            "rules_fired": [rule_fired] if rule_fired else [alert_type],
            "is_low_light": is_low_light,
            "thumbnail_url": f"/storage/thumbnails/{alert_id}.webp",
            "clip_url": f"/storage/clips/{alert_id}.mp4",
            "keyframe_webp_url": f"/storage/thumbnails/{alert_id}.webp",
            "clip_mp4_url": f"/storage/clips/{alert_id}.mp4",
            "clip_sha256": clip_sha,
            "notarization_token": token,
            "entity": {
                "entity_type": entity_log.entity_type,
                "upper_color": entity_log.upper_color,
                "lower_color": entity_log.lower_color,
                "posture": entity_log.posture,
                "plate_text": entity_log.plate_text,
                "location": {"lat": entity_log.location_lat, "lon": entity_log.location_lon}
            }
        }

        # Append to camera's cryptographic Merkle chain
        merkle_ledger.record_event(camera_id, alert_payload)

        # 4. Dispatch over MQTT broker
        self.publish_mqtt_event(settings.MQTT_TOPIC_ALERTS, alert_payload)

        # 5. Push to connected WebSocket clients
        from backend.app.api.ws_alerts import alert_ws_manager
        alert_ws_manager.broadcast_sync(alert_payload)

        return entity_log

alert_manager = AlertManager()
