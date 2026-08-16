-- SQLite schema. Same shape as sql/schema.sql, adjusted for SQLite types.
--
-- Applied automatically by store.open_store() when db.backend = sqlite, so there
-- is nothing to run by hand and no server to install.
--
-- Differences from the Postgres version, all mechanical:
--   SERIAL        -> INTEGER PRIMARY KEY AUTOINCREMENT
--   TIMESTAMPTZ   -> TEXT (ISO-8601 via CURRENT_TIMESTAMP)
--   JSONB         -> TEXT (json.dumps on the way in)
--   VECTOR(512)   -> BLOB (float32 bytes; ranked in NumPy, see store.LocalIndex)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS assets (
  asset_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  filename       TEXT NOT NULL UNIQUE,
  duration_s     REAL,
  fps            REAL,
  n_frames       INTEGER,
  width          INTEGER,
  height         INTEGER,
  codec          TEXT,
  proxy_path     TEXT,
  bytes          INTEGER,
  align_offset_s REAL,
  ingested_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
  event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  asset_id     INTEGER REFERENCES assets(asset_id) ON DELETE CASCADE,
  ext_event_id TEXT,
  event_type   TEXT,
  t_start_s    REAL NOT NULL,
  t_end_s      REAL NOT NULL,
  t_peak_s     REAL,
  confidence   REAL,
  n_players    INTEGER,
  pitch_zone   TEXT,
  source       TEXT DEFAULT 'model',
  UNIQUE (asset_id, ext_event_id)
);

CREATE TABLE IF NOT EXISTS clips (
  clip_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id   INTEGER UNIQUE REFERENCES events(event_id) ON DELETE CASCADE,
  path       TEXT NOT NULL,
  thumb_path TEXT,
  duration_s REAL,
  bytes      INTEGER
);

CREATE TABLE IF NOT EXISTS clip_embeddings (
  clip_id     INTEGER PRIMARY KEY REFERENCES clips(clip_id) ON DELETE CASCADE,
  embedding   BLOB NOT NULL,
  dim         INTEGER NOT NULL,
  model       TEXT,
  n_keyframes INTEGER,
  created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_log (
  query_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  q_text     TEXT,
  mode       TEXT,
  filters    TEXT,
  latency_ms REAL,
  encode_ms  REAL,
  search_ms  REAL,
  n_results  INTEGER,
  cold       INTEGER DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pilot_sessions (
  session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
  participant TEXT NOT NULL,
  started_at  TEXT DEFAULT CURRENT_TIMESTAMP,
  ended_at    TEXT,
  notes       TEXT
);

CREATE TABLE IF NOT EXISTS pilot_events (
  pe_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER REFERENCES pilot_sessions(session_id) ON DELETE CASCADE,
  task_id    TEXT,
  kind       TEXT,
  payload    TEXT,
  t_ms       REAL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS events_type_time_idx  ON events (event_type, t_start_s);
CREATE INDEX IF NOT EXISTS events_zone_idx       ON events (pitch_zone);
CREATE INDEX IF NOT EXISTS pilot_events_sess_idx ON pilot_events (session_id, task_id);

-- No ANN index. At single-match scale an exhaustive NumPy dot product over the
-- full embedding matrix is faster than any index probe, and s03 measures where
-- that stops being true.
