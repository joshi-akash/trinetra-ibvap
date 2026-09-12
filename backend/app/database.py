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

def init_db():
    from backend.app.models import User, CameraRegistry, EntityLog, FalseFlagLog, ExportLog, AuditLog
    Base.metadata.create_all(bind=engine)
    
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
                    geo_fence_polygon=[[78.1640, 29.9455], [78.1645, 29.9455], [78.1645, 29.9460], [78.1640, 29.9460]]
                ),
                CameraRegistry(
                    camera_id="CAM-02",
                    location_lat=29.9460,
                    location_lon=78.1650,
                    trust_score=0.95,
                    status="online"
                ),
                CameraRegistry(
                    camera_id="CAM-03",
                    location_lat=29.9450,
                    location_lon=78.1635,
                    trust_score=0.92,
                    status="online"
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
    finally:
        db.close()
