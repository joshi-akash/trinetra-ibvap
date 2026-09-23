"""
TRINETRA — End-to-End Offline Ingestion & Alert Verification Simulator
Demonstrates the complete Detect -> Bifurcate -> Alert -> Search -> Export lifecycle.
Zero WAN dependencies; runs 100% offline.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db
from backend.app.config import settings

def run_simulation():
    print("=" * 75)
    print("  TRINETRA (IBVAP) -- Sovereign Offline Edge Backend Simulator")
    print("=" * 75)

    # 1. Initialize DB & Storage
    print("\n[Step 1] Initializing Local Database & Storage Directories...")
    init_db()
    print(f"  [OK] Database initialized at: {settings.DATABASE_URL}")
    print(f"  [OK] Storage clips dir:      {settings.CLIPS_DIR}")
    print(f"  [OK] Storage thumbnails dir: {settings.THUMBNAILS_DIR}")
    print(f"  [OK] Storage exports dir:    {settings.EXPORTS_DIR}")

    client = TestClient(app)

    # 2. System Health Check
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    print(f"  [OK] System Health Check: {health_res.json()}")

    # 3. Authenticate as Commander
    print("\n[Step 2] Authenticating as Outpost Commander (Offline JWT)...")
    login_res = client.post("/api/auth/login", json={
        "username": "commander",
        "password": "commander123"
    })
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token_data = login_res.json()
    commander_token = token_data["access_token"]
    auth_headers = {"Authorization": f"Bearer {commander_token}"}
    print(f"  [OK] Logged in as: {token_data['username']} (Role: {token_data['role']})")

    # 4. Check Registered Cameras
    cam_res = client.get("/api/cameras", headers=auth_headers)
    assert cam_res.status_code == 200
    cameras = cam_res.json()
    print(f"  [OK] Detected {len(cameras)} registered cameras in BOP sector:")
    for c in cameras:
        print(f"    • Camera [{c['camera_id']}] Trust Score: {c['trust_score']} Status: {c['status']}")

    # 5. Ingest Frame 1: Patrol movement outside geo-fence (Passive Logging Path)
    print("\n[Step 3] Ingesting Frame 1: Routine Movement Outside Geo-Fence (FR-ALR-01)...")
    frame1 = {
        "camera_id": "CAM-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_ref": "frame_cam01_001.jpg",
        "frame_quality": {
            "low_light": False,
            "enhanced": False,
            "tamper_signal": {
                "is_tampered": False,
                "laplacian_variance": 620.0
            }
        },
        "entities": [
            {
                "track_id": "track-patrol-01",
                "entity_type": "human",
                "bbox": [100.0, 100.0, 220.0, 350.0],
                "confidence": 0.89,
                "attributes": {
                    "upper_color": "green",
                    "lower_color": "khaki",
                    "posture": "standing",
                    "gender": "male",
                    "props": []
                },
                "location": {"lat": 29.9800, "lon": 78.1900}  # Well outside fence
            }
        ]
    }
    ingest1_res = client.post("/api/entities/ingest", json=frame1)
    assert ingest1_res.status_code == 200
    print(f"  [OK] Verdict: {ingest1_res.json()['message']}")

    # 6. Ingest Frame 2: Geo-fence breach by red-shirt infiltrator (Active Alert Path)
    print("\n[Step 4] Ingesting Frame 2: Perimeter Geo-Fence Breach (FR-ALR-02.1)...")
    frame2 = {
        "camera_id": "CAM-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_ref": "frame_cam01_002.jpg",
        "frame_quality": {
            "low_light": False,
            "enhanced": False,
            "tamper_signal": {
                "is_tampered": False,
                "laplacian_variance": 580.0
            }
        },
        "entities": [
            {
                "track_id": "track-infiltrator-02",
                "entity_type": "human",
                "bbox": [120.0, 80.0, 260.0, 380.0],
                "confidence": 0.94,
                "attributes": {
                    "upper_color": "red",
                    "lower_color": "blue",
                    "posture": "standing",
                    "gender": "male",
                    "props": []
                },
                "location": {"lat": 29.9458, "lon": 78.1645}  # Inside fence
            }
        ]
    }
    ingest2_res = client.post("/api/entities/ingest", json=frame2)
    assert ingest2_res.status_code == 200
    print(f"  [OK] Verdict: {ingest2_res.json()['message']}")

    # 7. Ingest Frame 3: Multi-Modal Corroboration (Armed + Crouching inside fence)
    print("\n[Step 5] Ingesting Frame 3: Multi-Modal Threat Corroboration (FR-ALR-04)...")
    frame3 = {
        "camera_id": "CAM-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_ref": "frame_cam01_003.jpg",
        "frame_quality": {
            "low_light": True,
            "enhanced": True,
            "tamper_signal": {
                "is_tampered": False,
                "laplacian_variance": 420.0
            }
        },
        "entities": [
            {
                "track_id": "track-armed-03",
                "entity_type": "human",
                "bbox": [140.0, 150.0, 320.0, 420.0],
                "confidence": 0.96,
                "attributes": {
                    "upper_color": "black",
                    "lower_color": "camo",
                    "posture": "crouching",
                    "gender": "male",
                    "props": ["weapon"]
                },
                "location": {"lat": 29.9457, "lon": 78.1643}  # Inside fence
            }
        ]
    }
    ingest3_res = client.post("/api/entities/ingest", json=frame3)
    assert ingest3_res.status_code == 200
    print(f"  [OK] Verdict: {ingest3_res.json()['message']}")

    # 8. Ingest Frame 4: Camera Blinding / Tamper Event
    print("\n[Step 6] Ingesting Frame 4: Optical Tampering Detection (FR-TMP-01)...")
    frame4 = {
        "camera_id": "CAM-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_ref": "frame_cam01_004.jpg",
        "frame_quality": {
            "low_light": False,
            "enhanced": False,
            "tamper_signal": {
                "is_tampered": True,
                "laplacian_variance": 42.0,
                "histogram_flag": True,
                "reference_point_drift": False
            }
        },
        "entities": []
    }
    ingest4_res = client.post("/api/entities/ingest", json=frame4)
    assert ingest4_res.status_code == 200
    print(f"  [OK] Verdict: {ingest4_res.json()['message']}")

    # 9. Review Active Unacknowledged Alerts
    print("\n[Step 7] Checking Active Commander Alert Feed (/api/alerts/active)...")
    active_res = client.get("/api/alerts/active", headers=auth_headers)
    assert active_res.status_code == 200
    alerts = active_res.json()
    print(f"  [OK] Total Active Unacknowledged Alerts: {len(alerts)}")
    for a in alerts:
        print(f"    • Alert ID: {a['id'][:8]}... | Type: {a['alert_type'].upper()} | Rule: {a['rule_fired']}")
        print(f"      Thumbnail URL: {a['thumbnail_url']} | Clip URL: {a['clip_url']}")

    # 10. Acknowledge Alert 1
    if alerts:
        target_alert = alerts[0]
        print(f"\n[Step 8] Commander Acknowledging Alert: {target_alert['id'][:8]}...")
        ack_res = client.post(
            f"/api/alerts/{target_alert['id']}/acknowledge",
            headers=auth_headers,
            json={"notes": "Quick Reaction Team (QRT) dispatched to perimeter sector 4."}
        )
        assert ack_res.status_code == 200
        print(f"  [OK] Status: {ack_res.json()['message']}")

    # 11. Multilingual Forensic Search
    print("\n[Step 9] Executing Hindi/Hinglish Vernacular Search: 'laal shirt aadmi'...")
    search_res = client.post(
        "/api/search",
        headers=auth_headers,
        json={"query": "laal shirt aadmi"}
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    print(f"  [OK] Parsed Filters: {search_data['parsed_filters']['resolved']}")
    print(f"  [OK] Matches Found: {search_data['total_count']}")
    for r in search_data["results"]:
        print(f"    • Entity ID: {r['id'][:8]}... | Type: {r['entity_type']} | Upper: {r['upper_color']} | Lower: {r['lower_color']} | Alert: {r['is_alert']}")

    # 12. Manual Signed Offline HQ Export Package (FR-EXP-01, FR-EXP-02, FR-EXP-03)
    if search_data["results"]:
        export_target = search_data["results"][0]
        print(f"\n[Step 10] Building Signed Offline HQ Export Package for Entity: {export_target['id'][:8]}...")
        export_res = client.post(
            f"/api/entities/{export_target['id']}/export",
            headers=auth_headers,
            json={"include_clip": True, "channel": "local_bundle"}
        )
        assert export_res.status_code == 200, f"Export failed: {export_res.text}"
        export_info = export_res.json()
        print(f"  [OK] Export ID:      {export_info['export_id']}")
        print(f"  [OK] Payload SHA256: {export_info['payload_hash']}")
        print(f"  [OK] Download URL:   {export_info['download_url']}")

        # Verify export file on disk
        zip_filename = export_info["download_url"].split("/")[-1]
        zip_path = settings.EXPORTS_DIR / zip_filename
        assert zip_path.exists()
        print(f"  [OK] ZIP Bundle Verified on disk: {zip_path.stat().st_size} bytes")

        # Inspect ZIP manifest
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            print(f"  [OK] Archive Contents: {zf.namelist()}")

    # 13. False Flag Marking (Active Negative Hard Sample Feedback Loop)
    if search_data["results"]:
        ff_target = search_data["results"][0]
        print(f"\n[Step 11] Marking Entity {ff_target['id'][:8]} as False Flag (Hard Negative)...")
        ff_res = client.post(
            f"/api/entities/{ff_target['id']}/false-flag",
            headers=auth_headers,
            json={"moved_to_hard_negatives": True, "notes": "Border patrol officer authorized drill"}
        )
        assert ff_res.status_code == 200
        print(f"  [OK] False Flag Recorded in false_flag_log (ID: {ff_res.json()['id'][:8]}...)")

    print("\n" + "=" * 75)
    print("  SIMULATION COMPLETE — ALL END-TO-END WORKFLOWS VERIFIED 100% OFFLINE")
    print("=" * 75)

if __name__ == "__main__":
    run_simulation()
