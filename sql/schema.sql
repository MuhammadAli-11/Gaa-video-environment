-- gaa-video-environment schema
-- Mounted into the pgvector container at /docker-entrypoint-initdb.d, so it runs
-- automatically on first `docker compose up`. Safe to re-run by hand:
--   psql -h localhost -U postgres -d gaa -f sql/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Media assets
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
  asset_id      SERIAL PRIMARY KEY,
  filename      TEXT NOT NULL UNIQUE,
  duration_s    REAL,
  fps           REAL,
  n_frames      BIGINT,
  width         INT,
  height        INT,
  codec         TEXT,
  proxy_path    TEXT,
  bytes         BIGINT,
  align_offset_s REAL,          -- measured drift vs Project 1's manifest
  ingested_at   TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Events. `source` lets model-generated and hand-coded events live in one store,
-- which is the hybrid an applied analysis setup actually runs on.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
  event_id      SERIAL PRIMARY KEY,
  asset_id      INT REFERENCES assets(asset_id) ON DELETE CASCADE,
  ext_event_id  TEXT,              -- id carried over from Project 1, for traceability
  event_type    TEXT,              -- e.g. 'kickout_contest'
  t_start_s     REAL NOT NULL,
  t_end_s       REAL NOT NULL,
  t_peak_s      REAL,
  confidence    REAL,
  n_players     INT,
  pitch_zone    TEXT,              -- coarse: 'own_third' | 'middle' | 'opp_third'
  source        TEXT DEFAULT 'model',   -- 'model' | 'manual'
  UNIQUE (asset_id, ext_event_id)
);

CREATE TABLE IF NOT EXISTS clips (
  clip_id       SERIAL PRIMARY KEY,
  event_id      INT UNIQUE REFERENCES events(event_id) ON DELETE CASCADE,
  path          TEXT NOT NULL,
  thumb_path    TEXT,
  duration_s    REAL,
  bytes         BIGINT
);

-- Dimension must match embed.dim in config.yaml.
-- 512 = CLIP ViT-B/32. Change here and in config together if you swap encoder.
CREATE TABLE IF NOT EXISTS clip_embeddings (
  clip_id       INT PRIMARY KEY REFERENCES clips(clip_id) ON DELETE CASCADE,
  embedding     VECTOR(512) NOT NULL,
  dim           INT NOT NULL,
  model         TEXT,
  n_keyframes   INT,
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Query telemetry. Every /search hit lands here, which is what makes the
-- latency budget in the report measured rather than asserted.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_log (
  query_id      SERIAL PRIMARY KEY,
  q_text        TEXT,
  mode          TEXT,              -- 'semantic' | 'hybrid' | 'structured'
  filters       JSONB,
  latency_ms    REAL,
  encode_ms     REAL,
  search_ms     REAL,
  n_results     INT,
  cold          BOOLEAN DEFAULT FALSE,
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Pilot instrumentation. The UI posts here so time-on-task is measured by the
-- system rather than by someone holding a stopwatch.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pilot_sessions (
  session_id    SERIAL PRIMARY KEY,
  participant   TEXT NOT NULL,
  started_at    TIMESTAMPTZ DEFAULT now(),
  ended_at      TIMESTAMPTZ,
  notes         TEXT
);

CREATE TABLE IF NOT EXISTS pilot_events (
  pe_id         SERIAL PRIMARY KEY,
  session_id    INT REFERENCES pilot_sessions(session_id) ON DELETE CASCADE,
  task_id       TEXT,
  kind          TEXT,   -- task_start|query|clip_open|task_success|task_fail|task_abandon|note
  payload       JSONB,
  t_ms          REAL,   -- ms since task_start for this task
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS events_type_time_idx ON events (event_type, t_start_s);
CREATE INDEX IF NOT EXISTS events_zone_idx      ON events (pitch_zone);
CREATE INDEX IF NOT EXISTS pilot_events_sess_idx ON pilot_events (session_id, task_id);

-- The HNSW index is created here so a fresh database is immediately usable.
-- s03_build_index.py drops and rebuilds it in order to time the build honestly
-- and to report its storage footprint.
CREATE INDEX IF NOT EXISTS clip_embeddings_hnsw_idx
  ON clip_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
