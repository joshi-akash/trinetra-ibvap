"""
Persistent License Plate Tracker for TRINETRA ANPR.

Maintains temporal consistency of license plate reads across continuous
multi-frame vehicle trajectories, resolving partial OCR occlusions and transient glare.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .plate_ocr import clean_plate_text, validate_indian_plate


@dataclass
class PlateTrackState:
    track_id: Union[int, str]
    best_plate: Optional[str] = None
    best_confidence: float = 0.0
    history: List[Tuple[str, float]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    bbox: Optional[Tuple[int, int, int, int]] = None


class PlateTracker:
    """
    Temporal tracker caching and corroborating license plates across video tracks.
    """

    def __init__(self, max_history: int = 10, max_age_seconds: float = 5.0):
        self.max_history = max_history
        self.max_age_seconds = max_age_seconds
        self.tracks: Dict[Union[int, str], PlateTrackState] = {}

    def update_track(
        self,
        track_id: Union[int, str],
        raw_plate: Optional[str],
        confidence: float = 0.0,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[str]:
        """
        Update tracking state with a newly read plate candidate.
        Returns the best corroborated plate for this vehicle trajectory.
        """
        now = time.time()
        state = self.tracks.get(track_id)
        if state is None:
            state = PlateTrackState(track_id=track_id)
            self.tracks[track_id] = state

        state.last_updated = now
        if bbox:
            state.bbox = bbox

        if raw_plate:
            cleaned = clean_plate_text(raw_plate)
            is_valid = validate_indian_plate(cleaned)
            # Boost confidence for structurally valid Indian plates
            effective_conf = confidence * (1.2 if is_valid else 0.8)

            state.history.append((cleaned, effective_conf))
            if len(state.history) > self.max_history:
                state.history.pop(0)

            # Update best plate if higher confidence or first valid plate
            if (is_valid and state.best_plate is None) or (effective_conf > state.best_confidence):
                state.best_plate = cleaned
                state.best_confidence = effective_conf

        return state.best_plate

    def get_best_plate(self, track_id: Union[int, str]) -> Optional[str]:
        """Retrieve current highest-confidence plate for given track."""
        state = self.tracks.get(track_id)
        return state.best_plate if state else None

    def cleanup_stale(self, max_age: Optional[float] = None) -> None:
        """Purge trajectories that have left camera view."""
        age_limit = max_age or self.max_age_seconds
        now = time.time()
        stale_keys = [k for k, v in self.tracks.items() if (now - v.last_updated) > age_limit]
        for k in stale_keys:
            self.tracks.pop(k, None)

    def get_track_state(self, track_id: Union[int, str]) -> Optional[dict]:
        """Return the internal track state as a plain dict (for testing & diagnostics)."""
        state = self.tracks.get(track_id)
        if state is None:
            return None
        return {
            "track_id": state.track_id,
            "best_plate": state.best_plate,
            "best_confidence": state.best_confidence,
            "history": list(state.history),
            "last_updated": state.last_updated,
            "bbox": state.bbox,
        }

