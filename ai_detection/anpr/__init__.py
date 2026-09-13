"""ANPR (Automatic Number Plate Recognition) module for TRINETRA."""
from .plate_detector import PlateDetector
from .plate_ocr import PlateOCR, validate_indian_plate

__all__ = ["PlateDetector", "PlateOCR", "validate_indian_plate"]
