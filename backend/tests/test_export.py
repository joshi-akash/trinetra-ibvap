import pytest
import zipfile
import json
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models import EntityLog, User
from backend.app.export_service.export_builder import ExportService

@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed test user
    test_user = User(
        id="user-123",
        username="commander_test",
        password_hash="fakehash",
        role="commander"
    )
    session.add(test_user)

    # Seed test entity
    test_entity = EntityLog(
        id="entity-456",
        camera_id="CAM-01",
        timestamp=datetime.now(timezone.utc),
        entity_type="human",
        upper_color="blue",
        is_alert=True,
        alert_type="geo_fence",
        confidence_score=0.92,
        retention_tier="protected"
    )
    session.add(test_entity)
    session.commit()

    yield session
    session.close()

def test_export_bundle_generation(in_memory_db):
    user = in_memory_db.query(User).first()
    export_result = ExportService.build_export_package(
        db=in_memory_db,
        entity_id="entity-456",
        exporting_user=user,
        include_clip=False,
        channel="local_bundle"
    )

    assert "payload_hash" in export_result
    assert export_result["channel"] == "local_bundle"
    assert export_result["entity_log_id"] == "entity-456"

    # Verify that the generated zip file exists and contains valid metadata
    download_url = export_result["download_url"]
    filename = download_url.split("/")[-1]
    from backend.app.config import settings
    zip_path = settings.EXPORTS_DIR / filename
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        file_list = zf.namelist()
        assert "metadata.json" in file_list
        assert len(file_list) >= 2

        meta_bytes = zf.read("metadata.json")
        meta_data = json.loads(meta_bytes.decode("utf-8"))
        assert meta_data["record"]["id"] == "entity-456"
        assert meta_data["exported_by"]["username"] == "commander_test"

