from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings, PROJECT_ROOT
from backend.app.database import init_db
from backend.app.api.auth import router as auth_router
from backend.app.api.cameras import router as cameras_router
from backend.app.api.geofence import router as geofence_router
from backend.app.api.entities import router as entities_router
from backend.app.api.alerts import router as alerts_router
from backend.app.api.search import router as search_router
from backend.app.api.training import router as training_router
from backend.app.api.anpr import router as anpr_router
from backend.app.api.ws_alerts import ws_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database schema & seed data
    init_db()
    yield
    # Shutdown: Cleanup if needed

app = FastAPI(
    title="TRINETRA VA Edge Server",
    description="Offline-First Tactical AI-CCTV Video Analytics Backend & Infrastructure",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Storage Directories for Evidence Retrieval
app.mount("/storage/clips", StaticFiles(directory=str(settings.CLIPS_DIR)), name="clips")
app.mount("/storage/thumbnails", StaticFiles(directory=str(settings.THUMBNAILS_DIR)), name="thumbnails")
app.mount("/storage/exports", StaticFiles(directory=str(settings.EXPORTS_DIR)), name="exports")

# Mount test footage & uploaded CCTV videos for live playback
FOOTAGE_DIR = PROJECT_ROOT / "test_footage"
FOOTAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/footage", StaticFiles(directory=str(FOOTAGE_DIR)), name="footage")

from pathlib import Path
from fastapi.responses import FileResponse
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def get_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "TRINETRA VA Edge Server Running. Visit /docs for Swagger UI"}

# Register API & WebSocket Routers
app.include_router(auth_router)
app.include_router(cameras_router)
app.include_router(geofence_router)
app.include_router(entities_router)
app.include_router(alerts_router)
app.include_router(search_router)
app.include_router(training_router)
app.include_router(anpr_router)
app.include_router(ws_router)

@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "system": "TRINETRA",
        "sovereign_offline_ready": True
    }



