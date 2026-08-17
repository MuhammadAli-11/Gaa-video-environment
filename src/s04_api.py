"""s04_api.py — the retrieval service.

The design commitment: everything expensive happens at ingest. At query time the
only work is (a) encoding a short string with the CLIP text tower and (b) an ANN
lookup over pre-computed vectors. Nothing decodes video, nothing runs a detector,
nothing touches the GPU on the request path unless the text tower is on it.

Filters are applied as a SQL *pre-filter*, and the vector search then runs over the
reduced candidate set. This is the hybrid pattern, and it is here because pure
semantic search over sport video is unreliable: a general-purpose vision-language
model has no idea what a kickout is. The structured metadata from the vision layer
is what makes the retrieval usable. `mode=semantic` disables the pre-filter so the
two can be measured against each other — that comparison is the point.

Run:
    uvicorn s04_api:app --app-dir src --port 8000
    # or
    python src/s04_api.py

Then open http://127.0.0.1:8000/ for the UI.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from common import REPO_ROOT, dsn, load_config, pick_device, setup_logging

log = setup_logging("s04_api")
CFG = load_config(os.environ.get("GAA_CONFIG"))

# ---------------------------------------------------------------------------
# Text encoder — loaded lazily, once, behind a lock.
# Lazy so that cold-start latency is a thing we can measure rather than hide.
# ---------------------------------------------------------------------------
class TextEncoder:
    def __init__(self, cfg):
        self.cfg = cfg["embed"]
        self.device = pick_device(self.cfg.get("device", "auto"))
        self.template = self.cfg.get("text_prompt_template") or "{}"
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._lock = threading.Lock()
        self.load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> float:
        with self._lock:
            if self._model is not None:
                return self.load_seconds or 0.0
            import open_clip
            import torch

            t0 = time.perf_counter()
            model, _, _ = open_clip.create_model_and_transforms(
                self.cfg["model"], pretrained=self.cfg["pretrained"], device=self.device
            )
            model.eval()
            self._model = model
            self._tokenizer = open_clip.get_tokenizer(self.cfg["model"])
            self._torch = torch
            self.load_seconds = time.perf_counter() - t0
            log.info("Text tower ready on %s in %.2fs", self.device, self.load_seconds)
            return self.load_seconds

    def encode(self, text: str) -> np.ndarray:
        if os.environ.get("GAA_SMOKE_TEST"):
            # Deterministic pseudo-random vector keyed on the query string, so the
            # pipeline is exercisable without torch. Retrieval numbers produced this
            # way are noise; /health says so.
            seed = int.from_bytes(text.encode()[:8].ljust(8, b"\0"), "little") % (2**32)
            v = np.random.default_rng(seed).standard_normal(
                int(self.cfg["dim"])).astype(np.float32)
            return v / np.linalg.norm(v)
        if self._model is None:
            self.load()
        torch = self._torch
        prompt = self.template.format(text)
        tokens = self._tokenizer([prompt]).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].detach().cpu().numpy().astype(np.float32)


ENCODER = TextEncoder(CFG)
POOL = None
INDEX = None   # NumPy ranking path, sqlite backend only


@asynccontextmanager
async def lifespan(app: FastAPI):
    global POOL, INDEX
    import store

    if store.backend(CFG) == "sqlite":
        conn = store.init_sqlite(CFG)
        POOL = store.SqlitePool(conn)
        INDEX = store.LocalIndex().build(conn)
        log.info("SQLite store ready: %s (%d vectors loaded in %.1f ms).",
                 store.sqlite_path(CFG).name, INDEX.n, (INDEX.build_seconds or 0) * 1000)
    else:
        from psycopg_pool import ConnectionPool
        from pgvector.psycopg import register_vector

        ef = int(CFG.dot("index.ef_search", 40))

        def configure(conn):
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {ef}")

        POOL = ConnectionPool(dsn(CFG), min_size=2, max_size=10, configure=configure, open=True)
        POOL.wait(timeout=15)
        log.info("Postgres pool ready.")

    if CFG.dot("api.warmup_on_start", False):
        ENCODER.load()
        ENCODER.encode("warm up")
        log.info("Warm-up complete at startup.")
    else:
        log.info("Cold start mode: the text tower loads on the first /search. "
                 "POST /warmup before a pilot session.")
    yield
    POOL.close()


app = FastAPI(title="GAA video environment — retrieval layer", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log_query(q_text, mode, filters, latency_ms, encode_ms, search_ms, n_results, cold):
    try:
        with POOL.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO query_log (q_text, mode, filters, latency_ms, encode_ms,
                                          search_ms, n_results, cold)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (q_text, mode, json.dumps(filters), latency_ms, encode_ms,
                 search_ms, n_results, cold),
            )
    except Exception as exc:  # telemetry must never break the request path
        log.warning("query_log insert failed: %s", exc)


def rows_to_results(rows: list[tuple]) -> list[dict]:
    out = []
    for r in rows:
        (clip_id, event_id, event_type, t_start, t_end, t_peak, conf,
         n_players, zone, source, duration, score) = r
        out.append({
            "clip_id": clip_id,
            "event_id": event_id,
            "event_type": event_type,
            "t_start_s": round(t_start, 2) if t_start is not None else None,
            "t_end_s": round(t_end, 2) if t_end is not None else None,
            "t_peak_s": round(t_peak, 2) if t_peak is not None else None,
            "timecode": fmt_timecode(t_peak),
            "detector_confidence": round(conf, 3) if conf is not None else None,
            "n_players": n_players,
            "pitch_zone": zone,
            "source": source,
            "duration_s": round(duration, 2) if duration is not None else None,
            "similarity": round(float(score), 4) if score is not None else None,
            "clip_url": f"/clip/{clip_id}",
            "thumb_url": f"/thumb/{clip_id}",
        })
    return out


def fmt_timecode(t: float | None) -> str | None:
    if t is None:
        return None
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def build_filters(event_type, zone, min_confidence, t_from, t_to,
                  min_players=None) -> tuple[str, dict, dict]:
    clauses, params, applied = [], {}, {}
    if event_type:
        clauses.append("e.event_type = %(event_type)s")
        params["event_type"] = event_type
        applied["event_type"] = event_type
    if zone:
        clauses.append("e.pitch_zone = %(zone)s")
        params["zone"] = zone
        applied["pitch_zone"] = zone
    if min_confidence is not None and min_confidence > 0:
        clauses.append("COALESCE(e.confidence, 1.0) >= %(min_conf)s")
        params["min_conf"] = min_confidence
        applied["min_confidence"] = min_confidence
    if min_players is not None:
        # NULL is excluded rather than treated as zero. A clip whose player
        # count was never computed is not a clip with no players, and letting
        # NULL pass the filter would silently pad results with unmeasured clips.
        clauses.append("e.n_players IS NOT NULL AND e.n_players >= %(min_players)s")
        params["min_players"] = min_players
        applied["min_players"] = min_players
    if t_from is not None:
        clauses.append("e.t_peak_s >= %(t_from)s")
        params["t_from"] = t_from
        applied["t_from_s"] = t_from
    if t_to is not None:
        clauses.append("e.t_peak_s <= %(t_to)s")
        params["t_to"] = t_to
        applied["t_to_s"] = t_to
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params, applied


SELECT_COLS = """
    c.clip_id, e.event_id, e.event_type, e.t_start_s, e.t_end_s, e.t_peak_s,
    e.confidence, e.n_players, e.pitch_zone, e.source, c.duration_s
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness only. Deliberately does not trigger model loading, so that
    cold-start latency stays measurable. Use POST /warmup before a pilot."""
    db_ok, n_clips = False, None
    try:
        with POOL.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM clip_embeddings")
            n_clips = cur.fetchone()[0]
            db_ok = True
    except Exception as exc:  # noqa: BLE001
        log.warning("health db check failed: %s", exc)
    import store
    return {
        "status": "ok" if db_ok else "degraded",
        "backend": store.backend(CFG),
        "ranking": "numpy_exhaustive" if INDEX is not None else "pgvector_hnsw",
        "database": db_ok,
        "indexed_clips": n_clips,
        "model_loaded": ENCODER.loaded,
        "device": ENCODER.device,
        "model": f"{CFG.dot('embed.model')}/{CFG.dot('embed.pretrained')}",
        "smoke_test": bool(os.environ.get("GAA_SMOKE_TEST")),
    }


@app.post("/warmup")
def warmup():
    """Load the text tower and run one throwaway encode.

    In deployment this is what you call as the teams walk off at half-time, so the
    first real query is not the one that pays the model-load cost.
    """
    t0 = time.perf_counter()
    load_s = ENCODER.load()
    ENCODER.encode("a contested kickout")
    total = time.perf_counter() - t0
    return {"model_load_s": round(load_s, 3), "total_warmup_s": round(total, 3)}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="natural-language query"),
    event_type: str | None = None,
    zone: str | None = None,
    min_confidence: float | None = None,
    t_from: float | None = None,
    t_to: float | None = None,
    min_players: int | None = Query(
        default=None,
        description="minimum de-duplicated players in the contest window. "
                    "Clips with a NULL count are excluded, not treated as zero."),
    limit: int = Query(default=None),
    mode: Literal["hybrid", "semantic"] = "hybrid",
):
    """Semantic search over clip embeddings, with optional structured pre-filter.

    mode=hybrid   → SQL pre-filter narrows candidates, then cosine ANN ranks them.
    mode=semantic → filters ignored entirely; pure vector search. This is the
                    ablation arm for the retrieval evaluation.
    """
    limit = min(int(limit or CFG.dot("api.default_limit", 5)), int(CFG.dot("api.max_limit", 50)))
    cold = not ENCODER.loaded
    t_request = time.perf_counter()

    t0 = time.perf_counter()
    qvec = ENCODER.encode(q)
    encode_ms = (time.perf_counter() - t0) * 1000

    if mode == "hybrid":
        where, params, applied = build_filters(event_type, zone, min_confidence, t_from, t_to, min_players)
    else:
        where, params, applied = "", {}, {}

    t0 = time.perf_counter()
    if INDEX is not None:
        # sqlite backend: SQL narrows the candidates, NumPy ranks them.
        with POOL.connection() as conn, conn.cursor() as cur:
            if where:
                cur.execute(f"""SELECT c.clip_id FROM clips c
                                JOIN events e ON e.event_id = c.event_id {where}""", params)
                candidates = [r[0] for r in cur.fetchall()]
            else:
                candidates = None
            ranked = INDEX.search(qvec, limit, candidates)
            rows = []
            for clip_id, score in ranked:
                cur.execute(f"""SELECT {SELECT_COLS} FROM clips c
                                JOIN events e ON e.event_id = c.event_id
                                WHERE c.clip_id = %s""", (clip_id,))
                rows.append(tuple(cur.fetchone()) + (score,))
    else:
        # postgres backend: pgvector ranks inside the database.
        sql = f"""
            SELECT {SELECT_COLS}, 1 - (ce.embedding <=> %(qvec)s) AS similarity
            FROM clip_embeddings ce
            JOIN clips  c ON c.clip_id  = ce.clip_id
            JOIN events e ON e.event_id = c.event_id
            {where}
            ORDER BY ce.embedding <=> %(qvec)s
            LIMIT %(limit)s
        """
        params.update({"qvec": qvec, "limit": limit})
        with POOL.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    search_ms = (time.perf_counter() - t0) * 1000

    results = rows_to_results(rows)
    total_ms = (time.perf_counter() - t_request) * 1000
    log_query(q, mode, applied, total_ms, encode_ms, search_ms, len(results), cold)

    return {
        "query": q,
        "mode": mode,
        "filters_applied": applied,
        "n_results": len(results),
        "results": results,
        "timings_ms": {
            "text_encode": round(encode_ms, 2),
            "vector_search": round(search_ms, 2),
            "total": round(total_ms, 2),
        },
        "cold_start": cold,
    }


@app.get("/events")
def list_events(
    event_type: str | None = None,
    zone: str | None = None,
    min_confidence: float | None = None,
    t_from: float | None = None,
    t_to: float | None = None,
    min_players: int | None = Query(
        default=None,
        description="minimum de-duplicated players in the contest window. "
                    "Clips with a NULL count are excluded, not treated as zero."),
    limit: int = 50,
    order: Literal["time", "confidence"] = "time",
):
    """Structured filtering only. No embedding, no model — the fast path.

    Worth keeping separate: a lot of what an analyst wants at half-time is
    'show me every kickout in the first ten minutes', which is a WHERE clause, and
    routing that through a vision-language model would be slower and worse.
    """
    t_request = time.perf_counter()
    where, params, applied = build_filters(event_type, zone, min_confidence, t_from, t_to, min_players)
    order_by = "e.t_start_s ASC" if order == "time" else "e.confidence DESC NULLS LAST"
    sql = f"""
        SELECT {SELECT_COLS}, NULL AS similarity
        FROM events e
        JOIN clips c ON c.event_id = e.event_id
        {where}
        ORDER BY {order_by}
        LIMIT %(limit)s
    """
    params["limit"] = min(int(limit), 500)
    with POOL.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    total_ms = (time.perf_counter() - t_request) * 1000
    log_query(None, "structured", applied, total_ms, 0.0, total_ms, len(rows), False)
    return {
        "mode": "structured",
        "filters_applied": applied,
        "n_results": len(rows),
        "results": rows_to_results(rows),
        "timings_ms": {"total": round(total_ms, 2)},
    }


def _lookup_path(clip_id: int, column: str) -> Path:
    with POOL.connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {column} FROM clips WHERE clip_id = %s", (clip_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail=f"clip {clip_id} not found")
    path = Path(row[0])
    if not path.exists():
        raise HTTPException(status_code=410, detail=f"file missing on disk: {path.name}")
    return path


RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


@app.get("/clip/{clip_id}")
def stream_clip(clip_id: int, request: Request):
    """Serve a clip with HTTP range support.

    Range matters more than it looks: without 206 responses the browser downloads
    the whole file before it will let you seek, and the pilot showed that scrubbing
    backwards for context is the single most common interaction. Analysts scrub,
    they do not wait.
    """
    path = _lookup_path(clip_id, "path")
    size = path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            path, media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
        )

    match = RANGE_RE.match(range_header)
    if not match:
        raise HTTPException(status_code=416, detail="malformed Range header")
    g1, g2 = match.groups()
    if g1:
        start = int(g1)
        end = int(g2) if g2 else size - 1
    else:                              # suffix range: bytes=-N
        start = max(size - int(g2 or 0), 0)
        end = size - 1
    end = min(end, size - 1)
    if start > end:
        raise HTTPException(status_code=416, detail="range not satisfiable")
    length = end - start + 1

    def chunks(chunk_size: int = 1 << 16):
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                data = fh.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        chunks(), status_code=206, media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/thumb/{clip_id}")
def thumb(clip_id: int):
    return FileResponse(_lookup_path(clip_id, "thumb_path"), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/facets")
def facets():
    """Vocabulary the system actually understands, for the UI to display.

    Added after pilot participants repeatedly guessed at filter values. Showing
    the available terms is cheaper than teaching people to guess better.
    """
    with POOL.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT event_type FROM events WHERE event_type IS NOT NULL ORDER BY 1")
        types = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT pitch_zone FROM events WHERE pitch_zone IS NOT NULL ORDER BY 1")
        zones = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT count(*), COALESCE(max(t_end_s),0) FROM events")
        n_events, t_max = cur.fetchone()
    return {"event_types": types, "pitch_zones": zones,
            "n_events": n_events, "match_length_s": round(float(t_max), 1),
            "task_budget_s": CFG.dot("pilot.task_budget_s", 90)}


# ---------------------------------------------------------------------------
# Pilot telemetry
# ---------------------------------------------------------------------------
class SessionStart(BaseModel):
    participant: str
    notes: str | None = None


class PilotEvent(BaseModel):
    session_id: int
    task_id: str | None = None
    kind: str
    payload: dict[str, Any] | None = None
    t_ms: float | None = None


@app.post("/session/start")
def session_start(body: SessionStart):
    with POOL.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pilot_sessions (participant, notes) VALUES (%s,%s) RETURNING session_id",
            (body.participant, body.notes),
        )
        return {"session_id": cur.fetchone()[0]}


@app.post("/session/event")
def session_event(body: PilotEvent):
    with POOL.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO pilot_events (session_id, task_id, kind, payload, t_ms)
               VALUES (%s,%s,%s,%s,%s) RETURNING pe_id""",
            (body.session_id, body.task_id, body.kind,
             json.dumps(body.payload or {}), body.t_ms),
        )
        return {"pe_id": cur.fetchone()[0]}


@app.post("/session/{session_id}/end")
def session_end(session_id: int):
    with POOL.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE pilot_sessions SET ended_at = now() WHERE session_id = %s", (session_id,))
    return {"session_id": session_id, "ended": True}


# ---------------------------------------------------------------------------
# UI — mounted last so it does not shadow the API routes
# ---------------------------------------------------------------------------
UI_DIR = REPO_ROOT / "ui"
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "s04_api:app",
        host=str(CFG.dot("api.host", "127.0.0.1")),
        port=int(CFG.dot("api.port", 8000)),
        app_dir=str(REPO_ROOT / "src"),
        reload=False,
    )
