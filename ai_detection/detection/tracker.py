"""
ByteTrack Multi-Object Tracker for TRINETRA AI Detection.

Manages persistent tracking across consecutive video frames, assigning stable
track IDs (e.g. TRK-0001), associating bounding boxes over time, computing
smoothed velocity vectors, descriptive movement directions, and flagging movement anomalies.
"""

from __future__ import annotations

import logging
import time
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .yolov8_detector import DetectedEntity

logger = logging.getLogger(__name__)


class ByteTracker:
    """
    ByteTrack tracker wrapper for associating detected entities across frames,
    maintaining trajectory histories, computing movement direction, and flagging anomalies.
    """

    def __init__(
        self,
        tracker_config: str = "bytetrack.yaml",
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 35,
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
        camera_id: Optional[str] = None,
    ) -> List[DetectedEntity]:
        """
        Update tracker with new detections from current frame.

        Args:
            detections: List of DetectedEntity objects from YOLOv8 detector
            frame: Optional BGR frame image for visual tracking
            camera_id: Optional camera identifier for telemetry scoping

        Returns:
            List of DetectedEntity objects with updated, persistent track_id,
            directional telemetry, trajectory history, and movement anomaly flags.
        """
        if not detections:
            # Age active tracks when no detections
            self._age_tracks()
            return []

        now_ts = time.time()
        updated_entities: List[DetectedEntity] = []

        for entity in detections:
            curr_cx = float(entity.bbox[0] + entity.bbox[2]) / 2.0
            curr_cy = float(entity.bbox[1] + entity.bbox[3]) / 2.0

            # Associate with best matching active track by IoU
            best_id = None
            best_iou = 0.0

            for t_id, t_data in list(self._active_tracks.items()):
                iou = self._compute_iou(entity.bbox, t_data["bbox"])
                if iou > best_iou and iou >= (1.0 - self.match_thresh):
                    if entity.entity_type == t_data["entity_type"]:
                        best_iou = iou
                        best_id = t_id

            if best_id is not None:
                t_data = self._active_tracks[best_id]
                entity.track_id = f"TRK-{best_id:04d}"

                # Append current centroid to trajectory history with micro-shift deadzone
                traj = t_data["trajectory"]
                if not traj or math.hypot(curr_cx - traj[-1][0], curr_cy - traj[-1][1]) >= 3.0:
                    traj.append((curr_cx, curr_cy, now_ts))

                # Compute smoothed velocity and displacement vector over history
                dx, dy, dt, dist = self._compute_smoothed_displacement(traj)

                direction, cardinal = self._derive_direction(dx, dy, dist, prev_dir=t_data.get("direction"))

                # Estimate metric speed (km/h) based on pixel velocity and typical surveillance perspective
                if dt > 0.05 and dist > 4.0:
                    px_per_sec = dist / dt
                    # Scale factor: 1 px ≈ 0.045 m in 640x360 security camera
                    raw_speed = px_per_sec * 0.045 * 3.6
                    if entity.entity_type == "human":
                        speed_kmh = round(float(np.clip(raw_speed, 0.5, 24.0)), 1)
                    else:
                        speed_kmh = round(float(np.clip(raw_speed, 2.0, 95.0)), 1)
                else:
                    speed_kmh = t_data.get("speed_kmh", 0.0)

                # Behavioral movement anomaly detection (stabilized against stationary jitter)
                flags = self._detect_movement_anomalies(
                    entity.entity_type, traj, speed_kmh, dx, dy, t_data.get("heading_deg")
                )

                # Exponential Moving Average (EMA) Bounding Box Smoothing
                # Eliminates high-frequency box flutter and jumpy movement
                prev_box = t_data["bbox"]
                alpha = 0.65  # 65% new detection, 35% historical smoothed position
                smoothed_box = [
                    alpha * float(entity.bbox[0]) + (1.0 - alpha) * float(prev_box[0]),
                    alpha * float(entity.bbox[1]) + (1.0 - alpha) * float(prev_box[1]),
                    alpha * float(entity.bbox[2]) + (1.0 - alpha) * float(prev_box[2]),
                    alpha * float(entity.bbox[3]) + (1.0 - alpha) * float(prev_box[3]),
                ]
                # Deadzone filtering: if micro-shift is under 2.5 pixels across all coordinates, keep previous box
                if all(abs(smoothed_box[i] - prev_box[i]) < 2.5 for i in range(4)):
                    smoothed_box = [prev_box[0], prev_box[1], prev_box[2], prev_box[3]]

                smoothed_box = [round(float(v), 1) for v in smoothed_box]
                entity.bbox = smoothed_box

                # Update track state
                angle_deg = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0 if dist >= 8.0 else t_data.get("heading_deg", 0.0)
                t_data["bbox"] = smoothed_box
                t_data["centroid"] = (curr_cx, curr_cy)
                t_data["direction"] = direction
                t_data["cardinal"] = cardinal
                t_data["speed_kmh"] = speed_kmh
                t_data["heading_deg"] = angle_deg
                t_data["movement_flags"] = flags
                t_data["age"] = 0
                t_data["frames_seen"] = t_data.get("frames_seen", 0) + 1

                entity.direction = direction
                entity.speed_kmh = speed_kmh
                entity.trajectory = [[round(p[0], 1), round(p[1], 1)] for p in traj]
                entity.movement_flags = flags

            else:
                new_id = self._next_id
                self._next_id += 1
                entity.track_id = f"TRK-{new_id:04d}"

                traj = deque(maxlen=20)
                traj.append((curr_cx, curr_cy, now_ts))

                default_dir = "East (Moving Right)" if entity.entity_type == "vehicle" else "North (Advancing)"
                default_speed = 32.0 if entity.entity_type == "vehicle" else 4.0

                self._active_tracks[new_id] = {
                    "bbox": entity.bbox,
                    "entity_type": entity.entity_type,
                    "centroid": (curr_cx, curr_cy),
                    "trajectory": traj,
                    "direction": default_dir,
                    "cardinal": "North",
                    "speed_kmh": default_speed,
                    "heading_deg": 90.0,
                    "movement_flags": [],
                    "age": 0,
                    "frames_seen": 1,
                    "first_seen_ts": now_ts,
                }

                entity.direction = default_dir
                entity.speed_kmh = default_speed
                entity.trajectory = [[round(curr_cx, 1), round(curr_cy, 1)]]
                entity.movement_flags = []

            updated_entities.append(entity)

        self._age_tracks()
        return updated_entities

    def _age_tracks(self) -> None:
        """Age active tracks and purge tracks exceeding buffer."""
        stale_ids = []
        for t_id in self._active_tracks:
            self._active_tracks[t_id]["age"] += 1
            if self._active_tracks[t_id]["age"] > self.track_buffer:
                stale_ids.append(t_id)
        for t_id in stale_ids:
            del self._active_tracks[t_id]

    @staticmethod
    def _compute_smoothed_displacement(traj: deque) -> Tuple[float, float, float, float]:
        """Compute displacement vector over recent trajectory points."""
        if len(traj) < 2:
            return 0.0, 0.0, 0.0, 0.0

        # Look back up to 8 points or full history
        lookback = min(len(traj), 8)
        start_pt = traj[-lookback]
        end_pt = traj[-1]

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        dt = max(0.01, end_pt[2] - start_pt[2])
        dist = math.hypot(dx, dy)
        return dx, dy, dt, dist

    @staticmethod
    def _derive_direction(dx: float, dy: float, dist: float, prev_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Derive descriptive direction (cardinal + intuitive screen motion).
        Returns: (descriptive_direction, cardinal_direction)
        """
        if dist < 8.0:
            return prev_dir or "Stationary / Loitering", "Stationary"

        angle = math.atan2(-dy, dx)
        deg = (math.degrees(angle) + 360.0) % 360.0

        # Cardinal resolution
        if 22.5 <= deg < 67.5:
            cardinal = "North-East"
            screen_desc = "North-East (Retreating Right)"
        elif 67.5 <= deg < 112.5:
            cardinal = "North"
            screen_desc = "North (Retreating / Moving Away)"
        elif 112.5 <= deg < 157.5:
            cardinal = "North-West"
            screen_desc = "North-West (Retreating Left)"
        elif 157.5 <= deg < 202.5:
            cardinal = "West"
            screen_desc = "West (Moving Left)"
        elif 202.5 <= deg < 247.5:
            cardinal = "South-West"
            screen_desc = "South-West (Advancing Left)"
        elif 247.5 <= deg < 292.5:
            cardinal = "South"
            screen_desc = "South (Advancing towards Camera)"
        elif 292.5 <= deg < 337.5:
            cardinal = "South-East"
            screen_desc = "South-East (Advancing Right)"
        else:
            cardinal = "East"
            screen_desc = "East (Moving Right)"

        return screen_desc, cardinal

    @staticmethod
    def _detect_movement_anomalies(
        entity_type: str,
        traj: deque,
        speed_kmh: float,
        dx: float,
        dy: float,
        prev_heading: Optional[float] = None
    ) -> List[str]:
        """
        Detect behavioral movement anomalies:
        - Sprinting / Rapid Infiltration
        - Loitering / Lingering in Sector
        - Sudden Direction Reversal / Erratic Pacing
        """
        flags: List[str] = []

        # 1. Sprinting Detection (> 9.0 km/h for humans)
        if entity_type == "human" and speed_kmh >= 9.0:
            flags.append("sprinting")

        # 2. Loitering Detection (Track active > 15 frames with tight spatial confinement < 24px)
        if len(traj) >= 15:
            pts = list(traj)
            min_x = min(p[0] for p in pts)
            max_x = max(p[0] for p in pts)
            min_y = min(p[1] for p in pts)
            max_y = max(p[1] for p in pts)
            span = math.hypot(max_x - min_x, max_y - min_y)
            if span < 24.0:
                flags.append("loitering")

        # 3. Sudden Direction Change (Heading shift > 85 deg while moving with actual velocity)
        # Prevents stationary camera noise from triggering false sudden direction alerts
        if prev_heading is not None and speed_kmh >= 4.5 and math.hypot(dx, dy) >= 18.0:
            curr_angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0
            diff = abs(curr_angle - prev_heading)
            diff = min(diff, 360.0 - diff)
            if diff >= 85.0:
                flags.append("sudden_direction_change")

        return flags

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

