"""store.py — the storage layer, with two interchangeable backends.

    db.backend: sqlite      zero infrastructure. stdlib sqlite3 + a NumPy matrix.
    db.backend: postgres    Postgres 16 + pgvector + HNSW.

The rest of the codebase does not care which is in use. That is deliberate, and it
is the honest engineering answer at this scale rather than a convenience: with tens
or even thousands of clips, an exhaustive cosine over a contiguous float32 matrix is
a single BLAS call taking tens of microseconds. An HNSW probe cannot beat that,
and going out to a database over a socket to do it costs more in round-trip than the
arithmetic costs in total. The index is architecture for a season of footage, not
for a match, and `s03_build_index.py` measures where the crossover actually sits.

Two things this module provides:

  * A small shim so the same SQL strings work on both backends (`%s` placeholders,
    `ON CONFLICT DO UPDATE`, `RETURNING`, `now()`).
  * `LocalIndex`, the NumPy ranking path used by the sqlite backend.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

MIN_SQLITE = (3, 35, 0)  # RETURNING clause support

_NAMED = re.compile(r"%\((\w+)\)s")
_NOW = re.compile(r"\bnow\(\)", re.IGNORECASE)
_CAST = re.compile(r"::\s*(?:float|int|text|vector|numeric|real)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# SQL translation
# ---------------------------------------------------------------------------
def translate(sql: str) -> str:
    """Rewrite Postgres-flavoured SQL into the SQLite dialect.

    Only handles the constructs this codebase actually uses. It is not a general
    translator and is not trying to be.
    """
    sql = _NAMED.sub(r":\1", sql)
    sql = sql.replace("%s", "?")
    sql = _NOW.sub("CURRENT_TIMESTAMP", sql)
    sql = _CAST.sub("", sql)
    return sql


def adapt(params):
    """NumPy arrays become float32 BLOBs; dicts become JSON."""
    def one(v):
        if isinstance(v, np.ndarray):
            return sqlite3.Binary(np.asarray(v, dtype=np.float32).tobytes())
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        if isinstance(v, bool):
            return int(v)
        return v

    if params is None:
        return ()
    if isinstance(params, dict):
        return {k: one(v) for k, v in params.items()}
    return tuple(one(v) for v in params)


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# ---------------------------------------------------------------------------
# SQLite connection shim
# ---------------------------------------------------------------------------
class _CursorShim:
    def __init__(self, cur: sqlite3.Cursor):
        self._cur = cur

    def execute(self, sql, params=None):
        self._cur.execute(translate(sql), adapt(params))
        return self

    def executemany(self, sql, seq):
        self._cur.executemany(translate(sql), [adapt(p) for p in seq])
        return self

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def lastrowid(self):
        return self._cur.lastrowid

    @property
    def rowcount(self):
        return self._cur.rowcount


class SqliteConn:
    """Mimics the slice of the psycopg3 connection API this codebase uses."""

    backend = "sqlite"

    def __init__(self, path: Path):
        if sqlite3.sqlite_version_info < MIN_SQLITE:
            raise RuntimeError(
                f"SQLite {'.'.join(map(str, sqlite3.sqlite_version_info))} is too old; "
                f"{'.'.join(map(str, MIN_SQLITE))}+ is needed for RETURNING. "
                "Upgrade Python, or set db.backend: postgres in config.yaml."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        # One process-wide lock. The pilot has one facilitator and one participant;
        # contention is not a real concern and correctness is.
        self._lock = threading.RLock()

    @contextmanager
    def cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield _CursorShim(cur)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def close(self):
        self._conn.close()


class SqlitePool:
    """Pool-shaped wrapper so s04_api can treat both backends identically.

    SQLite connections are cheap and this one is shared, so the 'pool' is a single
    connection handed out under a lock. That is the correct design for an embedded
    database, not a compromise.
    """

    backend = "sqlite"

    def __init__(self, conn: SqliteConn):
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn

    def wait(self, timeout: float = 0):
        return True

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# The NumPy ranking path
# ---------------------------------------------------------------------------
class LocalIndex:
    """Exhaustive cosine ranking over an in-memory float32 matrix.

    Embeddings are L2-normalised at write time, so cosine similarity is a plain dot
    product and the whole search is `mat @ q` followed by an argpartition. Exact by
    construction: there is no recall/latency tradeoff to tune and no approximation
    error to report, which removes an entire class of caveat from the evaluation.
    """

    def __init__(self):
        self.ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.mat: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.row_of: dict[int, int] = {}
        self.built_at: float | None = None
        self.build_seconds: float | None = None
        self._lock = threading.RLock()

    @property
    def n(self) -> int:
        return int(self.ids.shape[0])

    def build(self, conn) -> "LocalIndex":
        with self._lock:
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute("SELECT clip_id, embedding, dim FROM clip_embeddings ORDER BY clip_id")
                rows = cur.fetchall()
            if not rows:
                self.ids = np.empty(0, dtype=np.int64)
                self.mat = np.empty((0, 0), dtype=np.float32)
                self.row_of = {}
            else:
                self.ids = np.array([r[0] for r in rows], dtype=np.int64)
                vecs = [blob_to_vec(r[1]) if isinstance(r[1], (bytes, memoryview))
                        else np.asarray(r[1], dtype=np.float32) for r in rows]
                dims = {v.shape[0] for v in vecs}
                if len(dims) > 1:
                    raise RuntimeError(f"mixed embedding dimensions in store: {sorted(dims)}")
                # C-contiguous so the matrix-vector product is one BLAS call.
                self.mat = np.ascontiguousarray(np.vstack(vecs), dtype=np.float32)
                # Defensive re-normalisation: the ranking is only a dot product if
                # every row is unit length, and a single bad row would rank high
                # against everything.
                norms = np.linalg.norm(self.mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.mat /= norms
                self.row_of = {int(cid): i for i, cid in enumerate(self.ids)}
            self.build_seconds = time.perf_counter() - t0
            self.built_at = time.time()
            return self

    def search(self, qvec: np.ndarray, k: int,
               candidate_ids: list[int] | None = None) -> list[tuple[int, float]]:
        """Return [(clip_id, cosine_similarity)] ranked, best first.

        `candidate_ids` is the SQL pre-filter result. Restricting the matrix before
        ranking is what makes this the hybrid path rather than post-filtering a
        semantic result set, which would silently drop relevant clips off the end.
        """
        with self._lock:
            if self.n == 0:
                return []
            q = np.asarray(qvec, dtype=np.float32).ravel()
            norm = np.linalg.norm(q)
            if norm:
                q = q / norm

            if candidate_ids is None:
                mat, ids = self.mat, self.ids
            else:
                rows = [self.row_of[c] for c in candidate_ids if c in self.row_of]
                if not rows:
                    return []
                idx = np.asarray(rows, dtype=np.int64)
                mat, ids = self.mat[idx], self.ids[idx]

            sims = mat @ q
            k = min(k, sims.shape[0])
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            return [(int(ids[i]), float(sims[i])) for i in top]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def sqlite_path(cfg) -> Path:
    raw = cfg.dot("db.sqlite_path", "data/gaa.db")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        from common import REPO_ROOT
        p = REPO_ROOT / p
    return p


def init_sqlite(cfg) -> SqliteConn:
    """Open the SQLite database, creating the schema if it is not there yet."""
    from common import REPO_ROOT

    conn = SqliteConn(sqlite_path(cfg))
    schema = (REPO_ROOT / "sql" / "schema_sqlite.sql").read_text()
    with conn.cursor() as cur:
        cur._cur.executescript(schema)
    return conn


def backend(cfg) -> str:
    return str(cfg.dot("db.backend", "sqlite")).lower()
