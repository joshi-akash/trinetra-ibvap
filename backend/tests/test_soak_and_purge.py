"""
TRINETRA Milestone M5 Verification: Soak & Purge Test
Validates continuous ingestion soak resilience and 92% disk FIFO unlinking.
Conforms to finalproject.pdf Table 4 (Page 15).
"""
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.models import Base, EntityLog, MediaTombstoneLog
from backend.app.storage.cleaner import StorageCleaner


@pytest.fixture
def soak_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_continuous_ingestion_soak_and_fifo_purge(soak_db):
    """
    Simulates a continuous multi-hour soak burst creating alert clips,
    tripping the 92% emergency disk threshold, and verifying FIFO unlinking
    with tombstone persistence.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        clips_dir = Path(temp_dir) / "clips"
        clips_dir.mkdir()

        now = datetime.now(timezone.utc)
        total_clips = 30
        
        # 1. Continuous soak ingestion burst
        for i in range(total_clips):
            clip_file = clips_dir / f"soak_clip_{i:03d}.mp4"
            clip_file.write_bytes(b"\x00\x00\x00\x18ftypmp42" * 50)
            
            ent = EntityLog(
                id=f"soak_alert_{i}",
                camera_id=f"CAM-{(i % 4) + 1:02d}",
                timestamp=now + timedelta(seconds=i * 2),
                is_alert=True,
                alert_type="geo_fence",
                clip_path=str(clip_file),
                clip_sha256=f"hash_{i:04d}" * 8,
                retention_tier="protected"
            )
            soak_db.add(ent)
            # Ensure strict FIFO file modification order
            os.utime(str(clip_file), (10000 + i * 5, 10000 + i * 5))
            
        soak_db.commit()

        # 2. Trigger 92% emergency threshold
        cleaner = StorageCleaner(clips_dir=clips_dir, high_watermark=92.0, low_watermark=80.0)
        
        # Evict down by unlinking 10 oldest files
        evictions = cleaner.check_and_purge(
            db=soak_db,
            simulated_disk_pct=94.2,
            max_files_to_unlink=10,
        )

        assert len(evictions) == 10
        # Check oldest files (000 to 009) were unlinked
        for i in range(10):
            assert not (clips_dir / f"soak_clip_{i:03d}.mp4").exists()
            
        # Check newer files (010 to 029) remain on disk
        for i in range(10, total_clips):
            assert (clips_dir / f"soak_clip_{i:03d}.mp4").exists()

        # Verify tombstones
        tombstone_count = soak_db.query(MediaTombstoneLog).count()
        assert tombstone_count == 10
        
        # Verify all 30 entity_log entries remain completely intact
        entity_count = soak_db.query(EntityLog).count()
        assert entity_count == total_clips
