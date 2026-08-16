"""s03_build_index.py — build the ranking index and measure what it actually costs.

The claim this script exists to test: **at single-match scale an ANN index is not
worth having.** Exhaustive cosine over an in-memory float32 matrix is one BLAS call;
HNSW cannot beat it, and a database round-trip to attempt it costs more than the
arithmetic. Rather than assert that, this measures it, and finds the corpus size at
which the answer flips.

sqlite backend   builds the NumPy matrix, times exhaustive search, and sweeps
                 synthetic corpus sizes to locate the crossover.
postgres backend additionally drops and rebuilds the HNSW index so build time and
                 storage footprint are real measurements, and reports the query plan.

Usage:
    python src/s03_build_index.py
    python src/s03_build_index.py --sweep 1000 10000 100000 1000000
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import store
from common import human_mb, load_config, connect, setup_logging, write_json

log = setup_logging("s03_build_index")

INDEX_NAME = "clip_embeddings_hnsw_idx"


def time_search(index: store.LocalIndex, n_probes: int, k: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal((n_probes, index.mat.shape[1])).astype(np.float32)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    times = []
    for p in probes:
        t0 = time.perf_counter()
        index.search(p, k)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def summarise(times: list[float]) -> dict:
    a = np.array(times)
    return {
        "n_probes": len(a),
        "mean_ms": round(float(a.mean()), 4),
        "p50_ms": round(float(np.percentile(a, 50)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "p99_ms": round(float(np.percentile(a, 99)), 4),
    }


def synthetic_sweep(dim: int, sizes: list[int], k: int, n_probes: int = 200) -> list[dict]:
    """Exhaustive NumPy search over progressively larger synthetic corpora.

    Random unit vectors are the pessimal case for an ANN index (no cluster structure
    to exploit), so treat the crossover this finds as a lower bound on where HNSW
    starts helping on real data.
    """
    rng = np.random.default_rng(42)
    out = []
    for n in sizes:
        # Fill float32 in chunks. `standard_normal((n, dim)).astype(np.float32)`
        # materialises the float64 array first, so n=1e6 needed 3.8 GiB to
        # produce a 1.9 GiB result and died on the allocation, not the workload.
        mat = np.empty((n, dim), dtype=np.float32)
        chunk = max(1, 2_000_000 // dim)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            block = rng.standard_normal((j - i, dim), dtype=np.float32)
            # normalise per chunk: np.linalg.norm over the whole matrix builds a
            # second full-size temporary, doubling peak memory for no reason
            block /= np.linalg.norm(block, axis=1, keepdims=True)
            mat[i:j] = block
        idx = store.LocalIndex()
        idx.mat = np.ascontiguousarray(mat)
        idx.ids = np.arange(n, dtype=np.int64)
        idx.row_of = {i: i for i in range(n)}
        s = summarise(time_search(idx, n_probes, k, seed=n))
        s.update({
            "n_vectors": n,
            "matrix_mb": human_mb(mat.nbytes),
            "within_500ms_budget": s["p95_ms"] < 500,
        })
        out.append(s)
        log.info("  n=%-9d p50 %8.3f ms | p95 %8.3f ms | %7.1f MB resident",
                 n, s["p50_ms"], s["p95_ms"], s["matrix_mb"])
    return out


def postgres_index(cfg, conn, dim: int) -> dict:
    m = int(cfg.dot("index.m", 16))
    efc = int(cfg.dot("index.ef_construction", 64))
    log.info("Rebuilding %s (m=%d, ef_construction=%d)…", INDEX_NAME, m, efc)
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
        t0 = time.perf_counter()
        cur.execute(
            f"CREATE INDEX {INDEX_NAME} ON clip_embeddings "
            f"USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {efc})"
        )
        build_s = time.perf_counter() - t0
        cur.execute("ANALYZE clip_embeddings")
        cur.execute("SELECT pg_relation_size('clip_embeddings'), "
                    "COALESCE(pg_relation_size(to_regclass(%s)), 0)", (INDEX_NAME,))
        tbl, idx = cur.fetchone()

    probe = np.random.default_rng(0).standard_normal(dim).astype(np.float32)
    probe /= np.linalg.norm(probe)
    with conn.cursor() as cur:
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) SELECT clip_id FROM clip_embeddings "
                    "ORDER BY embedding <=> %s LIMIT 5", (probe,))
        plan = cur.fetchone()[0][0]

    def walk(node):
        yield node.get("Node Type", "")
        for child in node.get("Plans", []) or []:
            yield from walk(child)

    node_types = list(walk(plan["Plan"]))
    return {
        "build_seconds": round(build_s, 4),
        "table_mb": human_mb(tbl),
        "index_mb": human_mb(idx),
        "execution_ms": round(plan.get("Execution Time", 0.0), 3),
        "node_types": node_types,
        "index_used": any("Index Scan" in t for t in node_types),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and measure the ranking index.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sweep", type=int, nargs="*",
                    default=[100, 1_000, 10_000, 100_000, 1_000_000])
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--probes", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dim = int(cfg.dot("embed.dim", 512))
    backend = store.backend(cfg)
    conn = connect(cfg)

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM clip_embeddings")
        n = cur.fetchone()[0]
    if n == 0:
        raise SystemExit("No embeddings found. Run: python src/s02_embed.py")

    index = store.LocalIndex().build(conn)
    log.info("Loaded %d vectors (dim %d) into a %.2f MB matrix in %.1f ms.",
             index.n, index.mat.shape[1], human_mb(index.mat.nbytes),
             (index.build_seconds or 0) * 1000)

    real = summarise(time_search(index, args.probes, args.k))
    log.info("Exhaustive search over the real corpus: p50 %.4f ms | p95 %.4f ms",
             real["p50_ms"], real["p95_ms"])

    stats = {
        "backend": backend,
        "n_vectors": index.n,
        "dim": int(index.mat.shape[1]),
        "matrix_mb": human_mb(index.mat.nbytes),
        "index_load_ms": round((index.build_seconds or 0) * 1000, 3),
        "exhaustive_search": real,
        "bytes_per_vector": int(index.mat.shape[1]) * 4,
    }

    if backend == "postgres":
        stats["postgres_hnsw"] = postgres_index(cfg, conn, dim)
        if not stats["postgres_hnsw"]["index_used"]:
            log.info("Planner chose a sequential scan — correct at %d vectors, and the "
                     "reason the sqlite backend gives up nothing here.", index.n)

    if not args.no_sweep:
        log.info("Sweeping synthetic corpus sizes…")
        stats["synthetic_sweep"] = synthetic_sweep(dim, args.sweep, args.k)
        over = [r["n_vectors"] for r in stats["synthetic_sweep"] if r["p95_ms"] > 10]
        stats["exhaustive_stays_under_10ms_up_to"] = (
            min(over) if over else f">{max(args.sweep)}"
        )
        log.info("Exhaustive NumPy search stays under 10 ms up to roughly %s vectors.",
                 stats["exhaustive_stays_under_10ms_up_to"])

    conn.close()
    out = write_json(cfg.get_path("paths.outputs_dir") / "index_stats.json", stats)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
