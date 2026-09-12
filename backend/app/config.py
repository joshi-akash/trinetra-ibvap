import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

class Settings(BaseSettings):
    APP_NAME: str = "TRINETRA Video Analytics Backend"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{BASE_DIR}/trinetra_local.db"
    )
    
    # MQTT
    MQTT_BROKER_HOST: str = os.getenv("MQTT_BROKER_HOST", "localhost")
    MQTT_BROKER_PORT: int = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    MQTT_TOPIC_ALERTS: str = "trinetra/alerts"
    MQTT_TOPIC_TELEMETRY: str = "trinetra/telemetry"
    
    # JWT Auth
    JWT_SECRET: str = os.getenv("JWT_SECRET", "trinetra_offline_jwt_secret_key_change_in_prod_32chars")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours for offline military shift
    
    # Storage
    STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))
    
    # Rule Engine / AI Thresholds
    THRESHOLDS_FILE: Path = Path(
        os.getenv("THRESHOLDS_FILE", str(PROJECT_ROOT / "ai_behavior" / "thresholds.yaml"))
    )

    # Ingestion & Buffer
    RING_BUFFER_DURATION_SEC: int = 60
    PRE_TRIGGER_DURATION_SEC: int = 15
    POST_TRIGGER_DURATION_SEC: int = 15
    
    @property
    def CLIPS_DIR(self) -> Path:
        p = self.STORAGE_DIR / "clips"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def THUMBNAILS_DIR(self) -> Path:
        p = self.STORAGE_DIR / "thumbnails"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def EXPORTS_DIR(self) -> Path:
        p = self.STORAGE_DIR / "exports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def RING_BUFFER_DIR(self) -> Path:
        p = self.STORAGE_DIR / "ring_buffer"
        p.mkdir(parents=True, exist_ok=True)
        return p

settings = Settings()
