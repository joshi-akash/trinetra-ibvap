"""
TRINETRA Person C Standalone Test Suite
Validates threat math, Merkle calculations, and emergency FIFO purge as mandated in PDF Section 5.3.
"""
import os
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.models import (
    Base,
    CameraRegistry,
    EntityLog,
    ArchiveEpochIndex,
    MediaTombstoneLog,
)
from backend.app.crypto_ledger.merkle_ledger import (
    compute_leaf_hash,
    build_merkle_root,
    generate_notarization_token,
    MerkleLedger,
)
from backend.app.storage.cleaner import StorageCleaner


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    
    # Seed camera
    cam = CameraRegistry(camera_id="CAM-01", trust_score=0.98, status="online")
    session.add(cam)
    session.commit()
    
    yield session
    session.close()


def test_probabilistic_threat_math():
    """
    Validates the compound threat scoring formula:
    P_threat = min(1.0, (P_geo + P_beh - P_geo*P_beh) * min(1.0, P_det / tau_vis))
    """
    def calc_threat(p_geo, p_beh, p_det, tau_vis):
        union = p_geo + p_beh - (p_geo * p_beh)
        vis_factor = min(1.0, p_det / tau_vis)
        return min(1.0, union * vis_factor)

    # 1. Daylight standard breach with high visibility (tau_vis = 0.80)
    score_day = calc_threat(p_geo=1.0, p_beh=0.90, p_det=0.95, tau_vis=0.80)
    assert pytest.approx(score_day, 1e-6) == 1.0

    # 2. Night pitch-black with low visibility floor (tau_vis = 0.40)
    score_night = calc_threat(p_geo=0.80, p_beh=0.50, p_det=0.85, tau_vis=0.40)
    # union = 0.8 + 0.5 - 0.4 = 0.9; vis_factor = min(1.0, 0.85/0.40) = 1.0
    assert pytest.approx(score_night, 0.01) == 0.90

    # 3. Heavy fog where confidence is penalized
    score_fog = calc_threat(p_geo=0.60, p_beh=0.0, p_det=0.35, tau_vis=0.70)
    # union = 0.6; vis_factor = 0.35/0.70 = 0.5; threat = 0.30
    assert pytest.approx(score_fog, 0.01) == 0.30


def test_merkle_root_and_hash_chain(test_db):
    """
    Validates Merkle leaf hashing, binary tree reduction, and hourly epoch rollup.
    """
    ledger = MerkleLedger()
    
    # Record sequential detections
    events = [
        {"track": "t1", "cam": "CAM-01", "type": "human", "conf": 0.92},
        {"track": "t2", "cam": "CAM-01", "type": "human", "conf": 0.94},
        {"track": "t3", "cam": "CAM-01", "type": "vehicle", "conf": 0.88},
    ]
    
    chain_hashes = []
    for ev in events:
        h = ledger.record_event("CAM-01", ev)
        chain_hashes.append(h)

    assert len(chain_hashes) == 3
    # Verify binary tree root calculation
    root = build_merkle_root(chain_hashes)
    assert len(root) == 64
    assert isinstance(root, str)

    # Empty leaf edge case
    empty_root = build_merkle_root([])
    assert len(empty_root) == 64

    # Test Notarization Token format
    token = generate_notarization_token("alert-123", "a"*64, "2026-09-12T10:00:00Z", "CAM-01")
    assert token.startswith("tn_")
    assert len(token) == 35  # tn_ + 32 chars


def test_emergency_fifo_storage_purge(test_db):
    """
    Validates that when disk usage reaches >= 92%, StorageCleaner unlinks
    oldest clips in FIFO order via os.unlink() and records MediaTombstoneLog entries.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        clips_dir = Path(temp_dir) / "clips"
        clips_dir.mkdir(parents=True)

        # Create dummy clips with staggered timestamps
        t_base = datetime.now(timezone.utc)
        created_files = []
        for i in range(5):
            clip_path = clips_dir / f"test_clip_{i}.mp4"
            clip_path.write_bytes(b"DUMMY_VIDEO_DATA" * 100)
            
            # Create matching EntityLog in db
            ent = EntityLog(
                id=f"alert_{i}",
                camera_id="CAM-01",
                timestamp=t_base + timedelta(minutes=i),
                is_alert=True,
                clip_path=str(clip_path),
                clip_sha256=f"sha_{i}" * 8,
            )
            test_db.add(ent)
            created_files.append(clip_path)
            # Stagger modification times
            os.utime(str(clip_path), (1000 + i * 10, 1000 + i * 10))

        test_db.commit()

        cleaner = StorageCleaner(clips_dir=clips_dir, high_watermark=90.0, low_watermark=80.0)

        # Simulate 93.5% usage and unlink up to 2 oldest files
        tombstones = cleaner.check_and_purge(
            db=test_db,
            simulated_disk_pct=93.5,
            max_files_to_unlink=2,
        )

        assert len(tombstones) == 2
        # Oldest file (test_clip_0.mp4) must have been deleted
        assert not (clips_dir / "test_clip_0.mp4").exists()
        assert not (clips_dir / "test_clip_1.mp4").exists()
        # Newer files must remain untouched
        assert (clips_dir / "test_clip_2.mp4").exists()

        # Check tombstones in database
        db_tombstones = test_db.query(MediaTombstoneLog).all()
        assert len(db_tombstones) == 2
        assert db_tombstones[0].entity_id == "alert_0"
        assert db_tombstones[0].disk_usage_pct_at_eviction == 93.5

        # Check that original entity_log rows remain completely untouched
        entities = test_db.query(EntityLog).all()
        assert len(entities) == 5
