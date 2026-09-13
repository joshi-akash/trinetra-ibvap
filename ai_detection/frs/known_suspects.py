"""
Known Suspects Local Database Store for TRINETRA FRS.

Provides local, fully offline storage and querying of enrolled known suspects
and their 512-dimensional normalized ArcFace biometric embeddings.

NOTE: The default runtime storage path 'ai_detection/data/suspects/' stays empty in this repo:
it is the live on-premise enrollment store, populated at deployment via Person C's admin-only
enrollment endpoint (/api/suspects/enroll), not seeded statically.
For benchmark evaluations and test suites, use 'ai_detection/data/suspects_bench/' (bench_mode=True).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default live runtime suspect store (empty in repo, populated via deployment enrollment)
DEFAULT_SUSPECTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "suspects"
)

# Benchmark fixture store for testing and offline latency evaluation
BENCH_SUSPECTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "suspects_bench"
)

SUSPECTS_METADATA_FILE = "suspects.json"
MATCH_SIMILARITY_THRESHOLD = 0.68  # FR-FRS-02 / Contract 1


@dataclass
class SuspectRecord:
    """Represents a single enrolled suspect."""
    suspect_id: str
    name: str
    embedding: List[float]  # 512-d ArcFace embedding
    notes: Optional[str] = None
    created_at: Optional[str] = None


class KnownSuspectStore:
    """
    Manages local storage and vector matching for enrolled suspects.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        bench_mode: bool = False,
    ):
        """
        Initialize KnownSuspectStore.

        Args:
            storage_dir: Custom storage path (if None, defaults to live or bench path)
            bench_mode: If True, uses 'data/suspects_bench/' fixture instead of live 'data/suspects/'
        """
        if storage_dir is not None:
            self.storage_dir = storage_dir
        elif bench_mode:
            self.storage_dir = BENCH_SUSPECTS_DIR
        else:
            self.storage_dir = DEFAULT_SUSPECTS_DIR

        self.suspects: Dict[str, SuspectRecord] = {}
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._id_list: List[str] = []

        os.makedirs(self.storage_dir, exist_ok=True)
        self.load_suspects()

    def _get_metadata_path(self) -> str:
        return os.path.join(self.storage_dir, SUSPECTS_METADATA_FILE)

    def load_suspects(self) -> None:
        """Load enrolled suspects from the local storage file."""
        meta_path = self._get_metadata_path()
        if not os.path.exists(meta_path):
            logger.info("No existing suspect database at %s. Initializing empty store.", meta_path)
            self.suspects = {}
            self._rebuild_index()
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.suspects = {
                    s_id: SuspectRecord(
                        suspect_id=s.get("suspect_id", s_id),
                        name=s.get("name", s_id),
                        embedding=s["embedding"],
                        notes=s.get("notes"),
                        created_at=s.get("created_at"),
                    )
                    for s_id, s in data.items()
                    if "embedding" in s
                }
            self._rebuild_index()
            logger.info("Loaded %d suspect(s) from %s", len(self.suspects), meta_path)
        except Exception as e:
            logger.error("Failed to load suspect database from %s: %s", meta_path, e)
            self.suspects = {}
            self._rebuild_index()

    def save_suspects(self) -> bool:
        """Persist enrolled suspects to the local JSON file."""
        meta_path = self._get_metadata_path()
        try:
            serializable = {
                s_id: {
                    "suspect_id": rec.suspect_id,
                    "name": rec.name,
                    "embedding": rec.embedding,
                    "notes": rec.notes,
                    "created_at": rec.created_at,
                }
                for s_id, rec in self.suspects.items()
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            return True
        except Exception as e:
            logger.error("Failed to save suspect database to %s: %s", meta_path, e)
            return False

    def _rebuild_index(self) -> None:
        """Rebuild internal normalized numpy embedding matrix for fast cosine similarity."""
        self._id_list = list(self.suspects.keys())
        if not self._id_list:
            self._embeddings_matrix = None
            return

        embeds = []
        for s_id in self._id_list:
            v = np.array(self.suspects[s_id].embedding, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                v = v / norm
            embeds.append(v)

        self._embeddings_matrix = np.vstack(embeds)  # Shape: (N, 512)

    def enroll_suspect(
        self,
        suspect_id: str,
        name: str,
        embedding: Union[np.ndarray, List[float]],
        notes: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> bool:
        """
        Enroll a new suspect with their 512-d normalized embedding.
        """
        if isinstance(embedding, np.ndarray):
            emb_vec = embedding.flatten().astype(np.float32)
        else:
            emb_vec = np.array(embedding, dtype=np.float32).flatten()

        if len(emb_vec) != 512:
            logger.error("Invalid embedding length %d for suspect %s. Expected 512.", len(emb_vec), suspect_id)
            return False

        # L2 normalize
        norm = np.linalg.norm(emb_vec)
        if norm > 1e-6:
            emb_vec = emb_vec / norm

        self.suspects[suspect_id] = SuspectRecord(
            suspect_id=suspect_id,
            name=name,
            embedding=emb_vec.tolist(),
            notes=notes,
            created_at=created_at,
        )
        self._rebuild_index()
        return self.save_suspects()

    def match_face(
        self,
        query_embedding: np.ndarray,
        threshold: float = MATCH_SIMILARITY_THRESHOLD,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute cosine similarity against all enrolled suspect embeddings.
        Returns {'suspect_id': str, 'confidence': float} if max similarity >= threshold,
        otherwise returns None.
        """
        if self._embeddings_matrix is None or len(self._id_list) == 0:
            return None

        q = query_embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            return None
        q = q / norm

        # Cosine similarity against all enrolled candidates
        similarities = np.dot(self._embeddings_matrix, q)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            matched_id = self._id_list[best_idx]
            return {
                "suspect_id": matched_id,
                "confidence": round(best_score, 4),
            }

        return None
