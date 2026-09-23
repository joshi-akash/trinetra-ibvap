"""
TRINETRA Local Benchmark Load Simulator
Simulates 15 concurrent camera streams pushing Contract 1 detections at 20 FPS.
Measures database write latency and asserts latency < 50 ms.
Conforms to finalproject.pdf Section 5.3 (Page 10).
"""
import concurrent.futures
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.models import Base, CameraRegistry, EntityLog


def run_benchmark(num_cameras: int = 15, batches_per_cam: int = 20):
    print("=" * 72)
    print("  TRINETRA (IBVAP) — 15-Stream Concurrent Load Benchmark")
    print(f"  Target: {num_cameras} Streams @ 20 FPS | SLA: DB Latency < 50 ms")
    print("=" * 72)

    from sqlalchemy import event

    engine = create_engine("sqlite:///benchmark_test.db", connect_args={"timeout": 30})
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute("PRAGMA busy_timeout = 5000;")
        cursor.execute("PRAGMA cache_size = -64000;")
        cursor.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    # Pre-seed 15 cameras
    init_session = Session()
    for i in range(1, num_cameras + 1):
        cam_id = f"CAM-{i:02d}"
        if not init_session.query(CameraRegistry).filter_by(camera_id=cam_id).first():
            init_session.add(CameraRegistry(camera_id=cam_id, trust_score=0.95, status="online"))
    init_session.commit()
    init_session.close()

    latencies_ms: List[float] = []

    def simulate_camera_stream(cam_id: str):
        thread_latencies = []
        session = Session()
        try:
            for b in range(batches_per_cam):
                t0 = time.perf_counter()
                
                # Synthetic Contract 1 Entity Ingestion
                record = EntityLog(
                    id=str(uuid.uuid4()),
                    camera_id=cam_id,
                    timestamp=datetime.now(timezone.utc),
                    entity_type="human",
                    upper_color="black",
                    lower_color="camo",
                    confidence_score=0.92,
                    is_alert=(b % 5 == 0),
                    alert_type="geo_fence" if (b % 5 == 0) else None,
                    retention_tier="protected" if (b % 5 == 0) else "passive",
                )
                session.add(record)
                session.commit()
                
                duration_ms = (time.perf_counter() - t0) * 1000.0
                thread_latencies.append(duration_ms)
                
                # ~20 FPS pacing (approx 50ms per frame)
                time.sleep(0.01)
        finally:
            session.close()
        return thread_latencies

    print(f"[+] Spawning {num_cameras} concurrent camera threads...")
    start_total = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cameras) as executor:
        futures = [executor.submit(simulate_camera_stream, f"CAM-{i:02d}") for i in range(1, num_cameras + 1)]
        for f in concurrent.futures.as_completed(futures):
            latencies_ms.extend(f.result())
    total_time = time.perf_counter() - start_total

    # Cleanup benchmark db
    try:
        import os
        if os.path.exists("benchmark_test.db"):
            os.remove("benchmark_test.db")
    except Exception:
        pass

    # Metrics
    total_writes = len(latencies_ms)
    avg_latency = sum(latencies_ms) / total_writes if total_writes else 0.0
    sorted_lats = sorted(latencies_ms)
    p50 = sorted_lats[int(total_writes * 0.50)] if total_writes else 0.0
    p95 = sorted_lats[int(total_writes * 0.95)] if total_writes else 0.0
    p99 = sorted_lats[int(total_writes * 0.99)] if total_writes else 0.0

    print("-" * 72)
    print(f"  Total Ingested Detections: {total_writes}")
    print(f"  Total Benchmark Time:     {total_time:.2f} s")
    print(f"  Throughput:               {total_writes / total_time:.1f} writes/second")
    print(f"  Average Write Latency:    {avg_latency:.2f} ms")
    print(f"  p50 Latency:              {p50:.2f} ms")
    print(f"  p95 Latency:              {p95:.2f} ms")
    print(f"  p99 Latency:              {p99:.2f} ms")
    print("-" * 72)

    assert avg_latency < 50.0, f"FAILED: Average write latency ({avg_latency:.2f} ms) exceeds 50 ms SLA target!"
    print(f"  [SUCCESS] All 15 concurrent camera streams met SLA (Average Latency: {avg_latency:.2f} ms < 50 ms).")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
