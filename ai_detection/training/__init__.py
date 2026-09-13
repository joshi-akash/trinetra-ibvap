"""Training module for TRINETRA AI Detection."""
from .train_yolo_border import train_border_model
from .export_tensorrt import export_to_tensorrt
from .dataset_downloader import generate_dataset_yaml, record_dataset_license
from .auto_trainer import AutoTrainer, get_auto_trainer, resolve_stream_source

__all__ = [
    "train_border_model",
    "export_to_tensorrt",
    "generate_dataset_yaml",
    "record_dataset_license",
    "AutoTrainer",
    "get_auto_trainer",
    "resolve_stream_source",
]
