# ==============================================================================
# TRINETRA IBVAP - Tactical AI-CCTV Sovereign Edge Server Dockerfile
# Production-Hardened Unified Container
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/app"

WORKDIR /app

# Install system dependencies for OpenCV, FFmpeg, GEOS, and video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgeos-dev \
    libpq-dev \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application layers
COPY ai_detection /app/ai_detection
COPY ai_behavior /app/ai_behavior
COPY backend /app/backend
COPY models /app/models
COPY test_footage /app/test_footage
COPY conftest.py /app/conftest.py

# Create required runtime storage and cache directories
RUN mkdir -p /app/backend/storage/clips \
             /app/backend/storage/thumbnails \
             /app/backend/storage/exports \
             /app/backend/storage/ring_buffer \
             /app/data/custom_dataset \
             /app/runs/train \
             /app/uploads/training_media

# Expose TRINETRA HTTP/WebSocket service port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Production entrypoint using Uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
