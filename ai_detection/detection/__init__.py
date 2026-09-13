"""Detection module for TRINETRA AI Detection."""
from .tracker import ByteTracker
from .yolov8_detector import DetectedEntity, YOLOv8Detector, compute_foot_point

__all__ = ["YOLOv8Detector", "ByteTracker", "DetectedEntity", "compute_foot_point"]
