from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.config import settings

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    connect_args["timeout"] = 15

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute("PRAGMA busy_timeout = 5000;")
        cursor.execute("PRAGMA cache_size = 10000;")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _auto_migrate_sqlite(db_engine):
    """Ensure newly added columns are present in existing SQLite tables."""
    if not str(db_engine.url).startswith("sqlite"):
        return
    import sqlite3
    raw_path = str(db_engine.url).replace("sqlite:///", "")
    try:
        conn = sqlite3.connect(raw_path)
        cur = conn.cursor()
        
        # camera_registry columns
        cur.execute("PRAGMA table_info(camera_registry)")
        cam_cols = {row[1] for row in cur.fetchall()}
        if "stream_url" not in cam_cols:
            cur.execute("ALTER TABLE camera_registry ADD COLUMN stream_url TEXT")
            
        # entity_log columns
        cur.execute("PRAGMA table_info(entity_log)")
        ent_cols = {row[1] for row in cur.fetchall()}
        for col, col_type in [
            ("vehicle_type", "TEXT"),
            ("direction", "TEXT"),
            ("speed_kmh", "REAL"),
            ("face_name", "TEXT"),
            ("skin_tone", "TEXT"),
            ("plate_thumbnail_path", "TEXT"),
        ]:
            if col not in ent_cols:
                cur.execute(f"ALTER TABLE entity_log ADD COLUMN {col} {col_type}")
                
        # Replace any obsolete cloud storage stream URLs with local offline presets
        cur.execute("UPDATE camera_registry SET stream_url = '/footage/highway_patrol.mp4' WHERE stream_url LIKE '%googleapis%'")
        conn.commit()
        conn.close()
    except Exception:
        pass

def init_db():
    from backend.app.models import User, CameraRegistry, EntityLog, FalseFlagLog, ExportLog, AuditLog, PlateWatchlist
    Base.metadata.create_all(bind=engine)
    _auto_migrate_sqlite(engine)
    
    # Auto-seed default cameras and users for instant local offline readiness
    db = SessionLocal()
    try:
        from backend.app.auth.jwt_auth import get_password_hash
        import uuid
        
        # Seed users if empty
        if db.query(User).count() == 0:
            default_users = [
                User(
                    id=str(uuid.UUID("a0000000-0000-0000-0000-000000000001")),
                    username="admin",
                    password_hash=get_password_hash("admin123"),
                    role="admin",
                    active=True
                ),
                User(
                    id=str(uuid.UUID("c0000000-0000-0000-0000-000000000001")),
                    username="commander",
                    password_hash=get_password_hash("commander123"),
                    role="commander",
                    active=True
                ),
                User(
                    id=str(uuid.UUID("e0000000-0000-0000-0000-000000000001")),
                    username="operator",
                    password_hash=get_password_hash("operator123"),
                    role="operator",
                    active=True
                ),
            ]
            db.add_all(default_users)
            db.commit()

        # Seed cameras if empty
        if db.query(CameraRegistry).count() == 0:
            default_cameras = [
                CameraRegistry(
                    camera_id="CAM-01",
                    location_lat=29.9457,
                    location_lon=78.1642,
                    trust_score=0.98,
                    status="online",
                    geo_fence_polygon=[[78.1640, 29.9455], [78.1645, 29.9455], [78.1645, 29.9460], [78.1640, 29.9460]],
                    stream_url="/footage/cctv_sample.mp4"
                ),
                CameraRegistry(
                    camera_id="CAM-02",
                    location_lat=29.9460,
                    location_lon=78.1650,
                    trust_score=0.95,
                    status="online",
                    stream_url="/footage/highway_patrol.mp4"
                ),
                CameraRegistry(
                    camera_id="CAM-03",
                    location_lat=29.9450,
                    location_lon=78.1635,
                    trust_score=0.92,
                    status="online",
                    stream_url="/footage/night_patrol.mp4"
                ),
                CameraRegistry(
                    camera_id="CAM-04",
                    location_lat=29.9470,
                    location_lon=78.1660,
                    trust_score=0.99,
                    status="online"
                ),
            ]
            db.add_all(default_cameras)
            db.commit()

        # Seed initial showcase intelligence entities & alerts if empty
        if db.query(EntityLog).count() == 0:
            import datetime
            now = datetime.datetime.now(datetime.timezone.utc)
            sample_entities = [
                EntityLog(
                    id=str(uuid.uuid4()),
                    camera_id="CAM-01",
                    timestamp=now - datetime.timedelta(minutes=2),
                    entity_type="human",
                    upper_color="black",
                    lower_color="navy_blue",
                    height_cm=174.5,
                    gender="male",
                    face_name="Suspect Alpha (SUSP-001)",
                    skin_tone="wheatish",
                    posture="crouching",
                    location_lat=29.9458,
                    location_lon=78.1643,
                    trajectory_id="TRK-0001",
                    is_alert=True,
                    alert_type="geo_fence",
                    confidence_score=0.94,
                    retention_tier="protected",
                    rule_fired="GEO-01: Restricted Perimeter Zone Infiltration",
                    direction="North",
                    speed_kmh=4.2
                ),
                EntityLog(
                    id=str(uuid.uuid4()),
                    camera_id="CAM-01",
                    timestamp=now - datetime.timedelta(minutes=5),
                    entity_type="vehicle",
                    vehicle_type="car",
                    upper_color="white",
                    plate_text="DL-01-AB-1234",
                    direction="North-East",
                    speed_kmh=42.5,
                    location_lat=29.9459,
                    location_lon=78.1646,
                    trajectory_id="TRK-0002",
                    is_alert=True,
                    alert_type="behavior",
                    confidence_score=0.91,
                    retention_tier="protected",
                    rule_fired="ANPR-02: Unregistered Vehicle High-Speed Approach"
                ),
                EntityLog(
                    id=str(uuid.uuid4()),
                    camera_id="CAM-02",
                    timestamp=now - datetime.timedelta(minutes=12),
                    entity_type="vehicle",
                    vehicle_type="truck",
                    upper_color="military_green",
                    plate_text="PB-08-XX-9901",
                    direction="North",
                    speed_kmh=34.0,
                    location_lat=29.9461,
                    location_lon=78.1652,
                    trajectory_id="TRK-0003",
                    is_alert=False,
                    confidence_score=0.88,
                    retention_tier="passive"
                ),
                EntityLog(
                    id=str(uuid.uuid4()),
                    camera_id="CAM-03",
                    timestamp=now - datetime.timedelta(minutes=18),
                    entity_type="human",
                    upper_color="gray",
                    lower_color="dark_brown",
                    height_cm=181.0,
                    gender="male",
                    face_name="Unidentified",
                    skin_tone="fair",
                    posture="standing",
                    location_lat=29.9452,
                    location_lon=78.1636,
                    trajectory_id="TRK-0004",
                    is_alert=False,
                    confidence_score=0.86,
                    retention_tier="passive",
                    direction="West",
                    speed_kmh=3.8
                ),
            ]
            db.add_all(sample_entities)
            db.commit()
    finally:
        db.close()
