-- TRINETRA Database Schema
-- Matches docs/00_INTEGRATION_CONTRACT.md §4 exactly

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Camera Registry
CREATE TABLE IF NOT EXISTS camera_registry (
  camera_id                     TEXT PRIMARY KEY,
  location                      GEOGRAPHY(Point),
  fov_polygon                   GEOGRAPHY(Polygon),
  geo_fence_polygon             GEOGRAPHY(Polygon),
  calibration_reference_points  JSONB,
  trust_score                   FLOAT DEFAULT 1.0,
  last_tamper_check             TIMESTAMPTZ DEFAULT now()
);

-- 2. Entity Log (Spine of the entire detection history)
CREATE TABLE IF NOT EXISTS entity_log (
  id                UUID PRIMARY KEY,
  camera_id         TEXT REFERENCES camera_registry(camera_id),
  timestamp         TIMESTAMPTZ NOT NULL,
  entity_type       TEXT,             -- human | vehicle | animal
  upper_color       TEXT,
  lower_color       TEXT,
  height_cm         FLOAT,
  gender            TEXT,             -- male | female | neutral
  plate_text        TEXT,
  location          GEOGRAPHY(Point),
  trajectory_id     UUID,
  is_alert          BOOLEAN DEFAULT FALSE,
  alert_type        TEXT,             -- geo_fence | behavior | tamper | correlated
  confidence_score  FLOAT,
  clip_path         TEXT,
  thumbnail_path    TEXT,
  retention_tier    TEXT DEFAULT 'passive', -- passive | protected
  deleted_manually  BOOLEAN DEFAULT FALSE,
  deletion_audit    JSONB,
  rule_fired        TEXT,
  acknowledged      BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_entity_timestamp ON entity_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_entity_is_alert ON entity_log(is_alert);
CREATE INDEX IF NOT EXISTS idx_entity_camera ON entity_log(camera_id);

-- 3. False Flag Log (Active feedback loop for model improvement)
CREATE TABLE IF NOT EXISTS false_flag_log (
  id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  entity_log_id            UUID REFERENCES entity_log(id) ON DELETE CASCADE,
  marked_by                TEXT,
  marked_at                TIMESTAMPTZ DEFAULT now(),
  moved_to_hard_negatives  BOOLEAN DEFAULT TRUE,
  notes                    TEXT
);

-- 4. Users Table (Local offline auth)
CREATE TABLE IF NOT EXISTS users (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  username      TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role          TEXT NOT NULL,        -- commander | operator | admin
  created_at    TIMESTAMPTZ DEFAULT now(),
  active        BOOLEAN DEFAULT TRUE
);

-- 5. Export Log (HQ Transport Audit Trail)
CREATE TABLE IF NOT EXISTS export_log (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  entity_log_id UUID REFERENCES entity_log(id),
  exported_by   UUID REFERENCES users(id),
  exported_at   TIMESTAMPTZ DEFAULT now(),
  payload_hash  TEXT,
  channel       TEXT
);

-- 6. Audit Log (Tracks critical operations: deletions, exports, permission changes)
CREATE TABLE IF NOT EXISTS audit_log (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id     UUID REFERENCES users(id),
  action      TEXT,
  target_id   UUID,
  timestamp   TIMESTAMPTZ DEFAULT now(),
  details     JSONB
);

-- Seed Default Cameras
INSERT INTO camera_registry (camera_id, location, geo_fence_polygon, trust_score)
VALUES 
  ('CAM-01', ST_SetSRID(ST_MakePoint(78.1642, 29.9457), 4326), NULL, 0.98),
  ('CAM-02', ST_SetSRID(ST_MakePoint(78.1650, 29.9460), 4326), NULL, 0.95),
  ('CAM-03', ST_SetSRID(ST_MakePoint(78.1635, 29.9450), 4326), NULL, 0.92),
  ('CAM-04', ST_SetSRID(ST_MakePoint(78.1660, 29.9470), 4326), NULL, 0.99)
ON CONFLICT (camera_id) DO NOTHING;

-- Seed Default Users (Password hashes are salted PBKDF2/SHA256 for admin123, commander123, operator123)
-- Admin
INSERT INTO users (id, username, password_hash, role)
VALUES (
  'a0000000-0000-0000-0000-000000000001',
  'admin',
  'pbkdf2:sha256:100000$trinetrasalt$c12c7d9954a6dbb7ec42a8b277d33d9f95701c3bf7ec84288005b8a6a61a6ec7',
  'admin'
) ON CONFLICT (username) DO NOTHING;

-- Commander
INSERT INTO users (id, username, password_hash, role)
VALUES (
  'c0000000-0000-0000-0000-000000000001',
  'commander',
  'pbkdf2:sha256:100000$trinetrasalt$be400ec4764b8ee76a9117cf4dd4e9cf76f7f2b1d6e1ea8dc1ee2f3c75ee93ff',
  'commander'
) ON CONFLICT (username) DO NOTHING;

-- Operator
INSERT INTO users (id, username, password_hash, role)
VALUES (
  'e0000000-0000-0000-0000-000000000001',
  'operator',
  'pbkdf2:sha256:100000$trinetrasalt$7f0556847844005ba3a789ef229ebce9c4e09f7a93070404481691a52f4eb60f',
  'operator'
) ON CONFLICT (username) DO NOTHING;
