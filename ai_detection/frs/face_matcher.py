"""
Face Matcher using InsightFace (RetinaFace + ArcFace) for TRINETRA FRS.

Enforces minimum resolution cutoff (>= 40x40 px) before embedding extraction,
computes 512-d normalized ArcFace embeddings, and matches against local suspects
with cosine similarity >= 0.68.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .known_suspects import KnownSuspectStore, MATCH_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

MIN_FACE_RESOLUTION = 40  # Minimum face width and height in pixels (40x40)


class FaceMatcher:
    """
    InsightFace-based Face Matcher with resolution gating and local store querying.
    """

    def __init__(
        self,
        suspect_store: Optional[KnownSuspectStore] = None,
        model_name: str = "buffalo_sc",
        similarity_threshold: float = MATCH_SIMILARITY_THRESHOLD,
        min_resolution: int = MIN_FACE_RESOLUTION,
        providers: Optional[List[str]] = None,
    ):
        self.suspect_store = suspect_store or KnownSuspectStore()
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_resolution = min_resolution
        self.app = None
        self._is_mock = False

        self._init_insightface(providers)

    def _init_insightface(self, providers: Optional[List[str]] = None) -> None:
        """Initialize InsightFace FaceAnalysis application."""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            prov = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.app = FaceAnalysis(name=self.model_name, providers=prov)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace (%s) initialized successfully.", self.model_name)
        except Exception as e:
            logger.warning("InsightFace initialization failed or not installed (%s). Using fallback mode.", e)
            self.app = None
            self._is_mock = True

    def extract_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512-d normalized ArcFace embedding from a face crop.
        Enforces resolution cutoff: face_crop must be at least min_resolution x min_resolution.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        h, w = face_crop.shape[:2]
        # Resolution gating: >= 40x40 px (FR-FRS-01)
        if h < self.min_resolution or w < self.min_resolution:
            logger.debug("Face crop resolution %dx%d below minimum cutoff %dx%d px. Skipping.", w, h, self.min_resolution, self.min_resolution)
            return None

        if self.app is not None and not self._is_mock:
            try:
                faces = self.app.get(face_crop)
                if not faces or len(faces) == 0:
                    return None

                # Find largest detected face
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embedding = largest_face.embedding.astype(np.float32)

                # L2 normalize
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    embedding = embedding / norm
                return embedding
            except Exception as e:
                logger.error("Error during face embedding extraction: %s", e)
                return None

        return None

    def process_person_crop(
        self,
        person_crop: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect face within a person crop, extract embedding if >= 40x40 px,
        and perform cosine matching against local suspect database.

        Returns:
            {'suspect_id': str, 'confidence': float} if matched >= 0.68,
            None otherwise (never fabricate a guess).
        """
        if person_crop is None or person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]
        if h < self.min_resolution or w < self.min_resolution:
            return None

        # If InsightFace is running, detect face on the person crop directly
        if self.app is not None and not self._is_mock:
            try:
                faces = self.app.get(person_crop)
                if not faces or len(faces) == 0:
                    return None

                for face in faces:
                    fx1, fy1, fx2, fy2 = face.bbox
                    fw = fx2 - fx1
                    fh = fy2 - fy1
                    # Gating: face resolution >= 40x40 px
                    if fw >= self.min_resolution and fh >= self.min_resolution:
                        emb = face.embedding.astype(np.float32)
                        norm = np.linalg.norm(emb)
                        if norm > 1e-6:
                            emb = emb / norm
                        return self.suspect_store.match_face(emb, threshold=self.similarity_threshold)
            except Exception as e:
                logger.error("Error matching face in person crop: %s", e)

        return None
