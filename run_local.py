#!/usr/bin/env python3
"""
TRINETRA (IBVAP) — Master Local Unified Platform Runner
Starts the complete sovereign offline video analytics platform locally on your device.
"""

import sys
import os
import time
import webbrowser
import threading
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_banner():
    banner = """
================================================================================
  TRINETRA (IBVAP) — Intelligent Border Video Analytics Platform
  Sovereign Offline Edge Node | BOP Sector Alpha
================================================================================
  • Architecture:  FastAPI + SQLite/PostGIS + Shapely + Mosquitto Bridge
  • Mode:          100% Offline by Design (Zero External Cloud / WAN)
  • Server URL:    http://localhost:8000
  • Dashboard:     http://localhost:8000/dashboard
  • API Docs:      http://localhost:8000/docs
  • Health Check:  http://localhost:8000/api/health
================================================================================
"""
    print(banner)

def open_browser():
    time.sleep(1.5)
    try:
        webbrowser.open("http://localhost:8000/")
    except Exception:
        pass

def main():
    print_banner()

    print("[1/3] Initializing local database, camera registry & audit ledger...")
    from backend.app.database import init_db
    init_db()
    print("      -> Database initialized and default cameras/users seeded successfully.")

    print("[2/3] Preparing live evidence storage directories...")
    from backend.app.config import settings
    settings.CLIPS_DIR
    settings.THUMBNAILS_DIR
    settings.EXPORTS_DIR
    print(f"      -> Storage ready at: {settings.STORAGE_DIR}")

    print("[3/3] Starting TRINETRA Edge Server on http://0.0.0.0:8000...")
    print("      -> Press CTRL+C to gracefully stop the outpost server.\n")

    # Launch browser automatically in background
    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
