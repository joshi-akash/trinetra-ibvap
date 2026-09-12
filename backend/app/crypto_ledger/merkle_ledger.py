"""
TRINETRA Cryptographic Merkle Ledger
Implements per-camera hash chains, hourly Merkle root rollups to archive_epoch_index,
and per-alert notarization tokens for evidential integrity defense in military/BSF custody.
Conforms to finalproject.pdf Item 1.1 & Section 5.3.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from backend.app.models import ArchiveEpochIndex, EntityLog


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def compute_leaf_hash(data: Any) -> str:
    """Computes a deterministic SHA-256 hash for an event or detection payload."""
    if isinstance(data, dict):
        # Canonical JSON string with sorted keys
        payload_bytes = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    elif isinstance(data, str):
        payload_bytes = data.encode("utf-8")
    elif isinstance(data, bytes):
        payload_bytes = data
    else:
        payload_bytes = str(data).encode("utf-8")
    return hashlib.sha256(payload_bytes).hexdigest()


def build_merkle_root(leaf_hashes: List[str]) -> str:
    """
    Constructs a standard binary Merkle tree from a list of leaf hashes (SHA-256 hex strings)
    and returns the 64-character Merkle root.
    Handles odd leaf counts by duplicating the last node.
    """
    if not leaf_hashes:
        return hashlib.sha256(b"TRINETRA_EMPTY_EPOCH_ROOT").hexdigest()

    current_layer = list(leaf_hashes)

    while len(current_layer) > 1:
        next_layer = []
        if len(current_layer) % 2 != 0:
            current_layer.append(current_layer[-1])

        for i in range(0, len(current_layer), 2):
            combined = current_layer[i] + current_layer[i + 1]
            parent_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
            next_layer.append(parent_hash)

        current_layer = next_layer

    return current_layer[0]


def generate_notarization_token(alert_id: str, clip_sha256: str, timestamp_iso: str, camera_id: str) -> str:
    """
    Generates a cryptographically-bound notarization token for an active alert.
    Binds alert UUID, video clip digest, timestamp, and camera identifier.
    """
    token_seed = f"TRINETRA_NOTARIZED:{alert_id}:{clip_sha256}:{timestamp_iso}:{camera_id}"
    token_hash = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()
    return f"tn_{token_hash[:32]}"


class MerkleLedger:
    """
    Stateful Merkle Ledger managing per-camera hash chains and
    rolling up epoch roots into archive_epoch_index.
    """
    def __init__(self):
        # camera_id -> list of leaf hashes accumulated in the active epoch
        self._active_epochs: Dict[str, List[str]] = {}
        # camera_id -> last chained hash
        self._chain_heads: Dict[str, str] = {}

    def record_event(self, camera_id: str, event_data: Dict[str, Any]) -> str:
        """
        Appends an event to the camera's active hash chain and epoch.
        Returns the chained event hash.
        """
        raw_leaf = compute_leaf_hash(event_data)
        prev_head = self._chain_heads.get(camera_id, "0" * 64)
        
        # Chained event hash: H(prev_head || raw_leaf)
        chained_hash = hashlib.sha256(f"{prev_head}:{raw_leaf}".encode("utf-8")).hexdigest()
        self._chain_heads[camera_id] = chained_hash

        if camera_id not in self._active_epochs:
            self._active_epochs[camera_id] = []
        self._active_epochs[camera_id].append(chained_hash)

        return chained_hash

    def rollup_epoch(
        self,
        db: Session,
        camera_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> ArchiveEpochIndex:
        """
        Calculates the Merkle root of all events in the epoch and records
        the rollup in archive_epoch_index.
        """
        # Fetch entities from DB for this epoch if in-memory list is empty
        leaves = self._active_epochs.get(camera_id, [])
        
        if not leaves:
            records = (
                db.query(EntityLog)
                .filter(
                    EntityLog.camera_id == camera_id,
                    EntityLog.timestamp >= start_time,
                    EntityLog.timestamp <= end_time,
                )
                .order_by(EntityLog.timestamp.asc())
                .all()
            )
            leaves = [
                compute_leaf_hash({
                    "id": r.id,
                    "cam": r.camera_id,
                    "ts": str(r.timestamp),
                    "alert": r.is_alert,
                    "type": r.entity_type,
                    "clip_sha": r.clip_sha256,
                })
                for r in records
            ]

        merkle_root = build_merkle_root(leaves)
        epoch_id = f"epoch_{camera_id}_{int(start_time.timestamp())}_{int(end_time.timestamp())}"

        # Count alerts
        alert_count = (
            db.query(EntityLog)
            .filter(
                EntityLog.camera_id == camera_id,
                EntityLog.timestamp >= start_time,
                EntityLog.timestamp <= end_time,
                EntityLog.is_alert == True,
            )
            .count()
        )

        epoch_record = ArchiveEpochIndex(
            id=str(uuid.uuid4()),
            epoch_id=epoch_id,
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            merkle_root=merkle_root,
            frame_count=len(leaves),
            alert_count=alert_count,
            created_at=utcnow(),
        )
        db.add(epoch_record)
        db.commit()
        db.refresh(epoch_record)

        # Clear active epoch cache for this camera
        self._active_epochs[camera_id] = []
        return epoch_record

    def verify_proof(self, leaf_hash: str, proof_path: List[Tuple[str, str]], root_hash: str) -> bool:
        """
        Verifies inclusion of leaf_hash in root_hash given proof_path pairs: (sibling_hash, direction)
        where direction is 'left' or 'right'.
        """
        current = leaf_hash
        for sibling, direction in proof_path:
            if direction == "left":
                combined = sibling + current
            else:
                combined = current + sibling
            current = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        return current == root_hash


# Singleton instance
merkle_ledger = MerkleLedger()
