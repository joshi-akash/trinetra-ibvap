FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for GEOS, OpenCV, FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/app /app/app
COPY ai_behavior/thresholds.yaml /app/ai_behavior/thresholds.yaml

# Ensure storage directories exist
RUN mkdir -p /app/storage/clips /app/storage/thumbnails /app/storage/exports /app/storage/ring_buffer

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
