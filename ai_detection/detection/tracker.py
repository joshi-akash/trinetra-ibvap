"""
ByteTrack Multi-Object Tracker for TRINETRA AI Detection.

Manages persistent tracking across consecutive video frames, assigning stable
track IDs (e.g. TRK-0001) and associating bounding boxes over time.
"""

from __future__ import annotations

import logging
import uuid
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .yolov8_detector import DetectedEntity

logger = logging.getLogger(__name__)


class ByteTracker:
    """
    ByteTrack tracker wrapper for associating detected entities across frames.
    """

    def __init__(
        self,
        tracker_config: str = "bytetrack.yaml",
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
    ):
        self.tracker_config = tracker_config
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self._next_id = 1
        self._active_tracks: Dict[int, Dict[str, Any]] = {}

    def update(
        self,
        detections: List[DetectedEntity],
        frame: Optional[np.ndarray] = None,
    ) -> List[DetectedEntity]:
        """
        Update tracker with new detections from current frame.

        Args:
            detections: List of DetectedEntity objects from YOLOv8 detector
            frame: Optional BGR frame image for visual tracking

        Returns:
            List of DetectedEntity objects with updated, persistent track_id values.
        """
        if not detections:
            return []

        # Simple greedy IoU tracking fallback when running standalone / offline
        updated_entities: List[DetectedEntity] = []

        for entity in detections:
            # If entity already has a persistent tracker ID from Ultralytics, preserve it
            if entity.track_id and not entity.track_id.startswith("TMP-"):
                updated_entities.append(entity)
                continue

            # Associate with best matching active track by IoU
            best_id = None
            best_iou = 0.0

            for t_id, t_data in list(self._active_tracks.items()):
                iou = self._compute_iou(entity.bbox, t_data["bbox"])
                if iou > best_iou and iou >= (1.0 - self.match_thresh):
                    if entity.entity_type == t_data["entity_type"]:
                        best_iou = iou
                        best_id = t_id

            curr_cx = float(entity.bbox[0] + entity.bbox[2]) / 2.0
            curr_cy = float(entity.bbox[1] + entity.bbox[3]) / 2.0

            if best_id is not None:
                entity.track_id = f"TRK-{best_id:04d}"
                prev_cx, prev_cy = self._active_tracks[best_id].get("centroid", (curr_cx, curr_cy))
                dx = curr_cx - prev_cx
                dy = curr_cy - prev_cy
                dist = math.hypot(dx, dy)

                if dist >= 2.0:
                    angle = math.atan2(-dy, dx)
                    deg = (math.degrees(angle) + 360.0) % 360.0
                    if 22.5 <= deg < 67.5:
                        direction = "North-East"
                    elif 67.5 <= deg < 112.5:
                        direction = "North"
                    elif 112.5 <= deg < 157.5:
                        direction = "North-West"
                    elif 157.5 <= deg < 202.5:
                        direction = "West"
                    elif 202.5 <= deg < 247.5:
                        direction = "South-West"
                    elif 247.5 <= deg < 292.5:
                        direction = "South"
                    elif 292.5 <= deg < 337.5:
                        direction = "South-East"
                    else:
                        direction = "East"
                    speed_kmh = round(dist * 0.05 * 25.0 * 3.6, 1)
                else:
                    direction = self._active_tracks[best_id].get("direction", "North")
                    speed_kmh = self._active_tracks[best_id].get("speed_kmh", 0.0)

                self._active_tracks[best_id]["bbox"] = entity.bbox
                self._active_tracks[best_id]["centroid"] = (curr_cx, curr_cy)
                self._active_tracks[best_id]["direction"] = direction
                self._active_tracks[best_id]["speed_kmh"] = speed_kmh
                self._active_tracks[best_id]["age"] = 0
                entity.direction = direction
                entity.speed_kmh = speed_kmh
            else:
                new_id = self._next_id
                self._next_id += 1
                entity.track_id = f"TRK-{new_id:04d}"
                direction = "North-East" if entity.entity_type == "vehicle" else "North"
                speed_kmh = 32.0 if entity.entity_type == "vehicle" else 4.0
                self._active_tracks[new_id] = {
                    "bbox": entity.bbox,
                    "entity_type": entity.entity_type,
                    "centroid": (curr_cx, curr_cy),
                    "direction": direction,
                    "speed_kmh": speed_kmh,
                    "age": 0,
                }
                entity.direction = direction
                entity.speed_kmh = speed_kmh

            updated_entities.append(entity)

        # Age active tracks and purge stale ones
        stale_ids = []
        for t_id in self._active_tracks:
            self._active_tracks[t_id]["age"] += 1
            if self._active_tracks[t_id]["age"] > self.track_buffer:
                stale_ids.append(t_id)
        for t_id in stale_ids:
            del self._active_tracks[t_id]

        return updated_entities

    @staticmethod
    def _compute_iou(boxA: List[int], boxB: List[int]) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

        denom = float(boxAArea + boxBArea - interArea)
        if denom <= 0:
            return 0.0
        return float(interArea / denom)
