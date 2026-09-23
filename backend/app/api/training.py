"""
API Endpoints for Continuous AI Model Training & Fine-Tuning.

Supports:
1. Ingesting uploaded CCTV video files (MP4, MKV, AVI, etc.) or stream URLs (YouTube Live, HLS, RTSP).
2. Spawning background frame extraction, pseudo-labeling, and Ultralytics YOLOv8 fine-tuning.
3. Telemetry polling (status, epochs, box/class loss, bounding box counts, logs).
4. Hot-reloading active detector singleton with refined custom weights.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.auth.jwt_auth import get_current_user
from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import (
    ApplyWeightsRequest,
    GenericStatusResponse,
    TrainingStartRequest,
    TrainingStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["Model Training"])


@router.post("/start-video-training")
async def start_video_training(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Initiate continuous model fine-tuning pipeline.
    Accepts both multipart/form-data (video file upload or form fields)
    and application/json (direct stream URL).
    """
    from ai_detection.training import get_auto_trainer

    trainer = get_auto_trainer()
    content_type = request.headers.get("content-type", "")

    source_type = "stream"
    source_path_or_url = ""
    epochs = 3
    max_frames = 40
    base_weights = "models/yolov8n.pt"
    enhance_low_light = True

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        stream_url = form.get("stream_url")
        epochs = int(form.get("epochs", 3))
        max_frames = int(form.get("max_frames", 40))
        base_weights = str(form.get("base_weights", "models/yolov8n.pt"))
        enhance_str = str(form.get("enhance_low_light", "true")).lower()
        enhance_low_light = enhance_str in ["true", "1", "yes"]

        if uploaded_file and hasattr(uploaded_file, "filename") and uploaded_file.filename:
            # Save uploaded video file
            upload_dir = Path("uploads/training_media")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_ext = Path(uploaded_file.filename).suffix or ".mp4"
            dest_path = upload_dir / f"train_cctv_{int(time.time())}{file_ext}"

            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)

            source_type = "file"
            source_path_or_url = str(dest_path.resolve())
            logger.info(f"Saved uploaded training video to: {source_path_or_url}")
        elif stream_url and str(stream_url).strip():
            source_type = "stream"
            source_path_or_url = str(stream_url).strip()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either a video file or stream_url must be provided for training.",
            )
    else:
        # JSON Payload
        try:
            body = await request.json()
        except Exception:
            body = {}
        stream_url = body.get("stream_url")
        epochs = int(body.get("epochs", 3))
        max_frames = int(body.get("max_frames", 40))
        base_weights = str(body.get("base_weights", "models/yolov8n.pt"))
        enhance_low_light = bool(body.get("enhance_low_light", True))

        if not stream_url or not str(stream_url).strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Stream URL is required in JSON payload.",
            )
        source_type = "stream"
        source_path_or_url = str(stream_url).strip()

    # Launch background training
    started = trainer.start_training(
        source_type=source_type,
        source_path_or_url=source_path_or_url,
        epochs=epochs,
        max_frames=max_frames,
        base_weights=base_weights,
        enhance_low_light=enhance_low_light,
    )

    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A model training session is already in progress. Please wait for completion.",
        )

    return {
        "status": "success",
        "message": "AI continual fine-tuning pipeline launched successfully.",
        "config": {
            "source_type": source_type,
            "epochs": epochs,
            "max_frames": max_frames,
            "base_weights": base_weights,
            "enhance_low_light": enhance_low_light,
        }
    }


@router.get("/status", response_model=TrainingStatusResponse)
def get_training_status(current_user: User = Depends(get_current_user)):
    """Retrieve real-time telemetry from active or last training run."""
    from ai_detection.training import get_auto_trainer
    trainer = get_auto_trainer()
    return trainer.get_status()


@router.post("/apply-weights", response_model=GenericStatusResponse)
def apply_custom_weights(
    req: Optional[ApplyWeightsRequest] = None,
    current_user: User = Depends(get_current_user),
):
    """Hot-reload the live detector engine with newly trained custom weights."""
    from ai_detection.training import get_auto_trainer
    trainer = get_auto_trainer()
    weights_path = req.weights_path if req and req.weights_path else None

    success = trainer.apply_weights(weights_path=weights_path)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not apply weights. Check if weights file exists.",
        )

    return GenericStatusResponse(
        status="success",
        message="Active YOLOv8 detection model hot-reloaded successfully with custom weights.",
    )


@router.get("/curated-datasets")
def get_curated_datasets(current_user: User = Depends(get_current_user)):
    """Retrieve list of pre-configured and benchmark surveillance datasets."""
    from ai_detection.training.dataset_downloader import CURATED_DATASETS
    return {
        "status": "success",
        "datasets": list(CURATED_DATASETS.values())
    }


@router.post("/load-preset-dataset")
async def load_preset_dataset(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Load or generate a curated surveillance dataset ready for model training."""
    from ai_detection.training.dataset_downloader import (
        CURATED_DATASETS,
        prepare_starter_surveillance_dataset,
        prepare_cctv_elevated_dataset,
    )

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    preset_id = body.get("preset_id", "cctv_elevated_pedestrians")

    if preset_id not in CURATED_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset preset: {preset_id}")

    preset = CURATED_DATASETS[preset_id]
    if preset_id == "cctv_elevated_pedestrians":
        res = prepare_cctv_elevated_dataset()
        return {
            "status": "success",
            "message": "Loaded and generated Overhead & Elevated CCTV Pedestrian dataset.",
            "dataset": res
        }
    elif preset_id == "starter_surveillance":
        res = prepare_starter_surveillance_dataset()
        return {
            "status": "success",
            "message": "Loaded and verified TRINETRA starter surveillance dataset.",
            "dataset": res
        }
    else:
        return {
            "status": "info",
            "message": f"Dataset '{preset['name']}' requires external download from {preset.get('source', '')}. Use dataset_downloader CLI or configure Roboflow/Kaggle credentials.",
            "dataset": preset
        }


@router.get("/multi-status")
def get_multi_model_status():
    """Retrieve real-time telemetry for all models in the batch training pipeline."""
    from ai_detection.training.multi_model_trainer import get_multi_trainer
    trainer = get_multi_trainer()
    return trainer.get_status()


@router.post("/start-all-downloaded")
def start_all_downloaded_training():
    """Launch simultaneous/sequential fine-tuning for all 5 surveillance fleet models."""
    from ai_detection.training.multi_model_trainer import get_multi_trainer
    trainer = get_multi_trainer()
    success = trainer.start_batch_training()
    if not success:
        return {
            "status": "conflict",
            "message": "A multi-model batch training session is already in progress.",
            "data": trainer.get_status(),
        }
    return {
        "status": "success",
        "message": "Batch retraining pipeline launched for all 5 surveillance models.",
        "data": trainer.get_status(),
    }


@router.post("/train-single")
async def train_single_model(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Launch dedicated fine-tuning for a specific surveillance model from the 5-model fleet."""
    from ai_detection.training.multi_model_trainer import get_multi_trainer
    trainer = get_multi_trainer()
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    model_id = body.get("model_id")
    epochs = body.get("epochs")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id parameter is required (e.g. tactical_weapons, night_vision, patrol_vehicles, visdrone_overhead, elevated_cctv)")

    # Validate model_id before launching — reject unknown models immediately
    VALID_MODEL_IDS = {"elevated_cctv", "patrol_vehicles", "tactical_weapons", "night_vision", "visdrone_overhead"}
    if str(model_id) not in VALID_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id '{model_id}'. Must be one of: {', '.join(sorted(VALID_MODEL_IDS))}"
        )

    success = trainer.start_single_model_training(
        model_id=str(model_id),
        epochs=int(epochs) if epochs else None,
    )
    if not success:
        raise HTTPException(
            status_code=409,
            detail=f"Training for model '{model_id}' is already in progress. Check /api/training/status."
        )
    return {
        "status": "success",
        "message": f"Dedicated training pipeline started for model: {model_id}",
        "data": trainer.get_status(),
    }



