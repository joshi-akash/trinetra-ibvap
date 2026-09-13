#!/usr/bin/env python3
"""
TRINETRA (IBVAP) — Master Local Unified Platform Runner
Starts the complete sovereign offline video analytics platform locally on your device
with dual-stack IPv4/IPv6 support for seamless localhost & 127.0.0.1 browser resolution.
"""

import sys
import os
import time
import socket
import asyncio
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
  • Primary URL:   http://127.0.0.1:8000/dashboard
  • Localhost URL: http://localhost:8000/dashboard
  • API Docs:      http://127.0.0.1:8000/docs
  • Health Check:  http://127.0.0.1:8000/api/health
================================================================================
"""
    print(banner)


def open_browser():
    time.sleep(1.2)
    for url in ["http://127.0.0.1:8000/dashboard", "http://localhost:8000/dashboard"]:
        try:
            webbrowser.open(url)
            break
        except Exception:
            pass


def create_dual_stack_socket(port: int = 8000):
    """
    Create a dual-stack socket listening on both IPv6 (::) and IPv4 (0.0.0.0)
    so modern Windows browsers (Edge/Chrome) resolving 'localhost' to ::1 or 127.0.0.1
    never receive ERR_CONNECTION_REFUSED.
    """
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('::', port))
        sock.listen(256)
        return sock
    except Exception:
        # Fallback to standard IPv4 if IPv6 stack is disabled
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(256)
        return sock


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

    print("[3/3] Starting TRINETRA Edge Server (Dual-Stack IPv4/IPv6 on Port 8000)...")
    print("      -> Accessible at http://127.0.0.1:8000/ and http://localhost:8000/")
    print("      -> Press CTRL+C to gracefully stop the outpost server.\n")

    # Launch browser automatically in background
    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    sock = create_dual_stack_socket(8000)
    config = uvicorn.Config("backend.app.main:app", log_level="info", reload=False)
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=[sock]))


if __name__ == "__main__":
    main()
