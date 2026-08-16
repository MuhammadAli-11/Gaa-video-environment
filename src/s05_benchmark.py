"""s05_benchmark.py — measure the thing the project claims to optimise.

Four modes:

  latency    warm p50/p95/p99, component breakdown (text encode vs ANN search),
             cold start measured by launching a fresh server process, playback
             time-to-first-byte, ingest throughput, storage per match-minute.
  retrieval  precision@k / recall@10 / MRR / nDCG@10 over a hand-judged query set,
             computed twice — pure semantic and hybrid — because the delta between
             them is the architectural finding.
  scale      synthetic vectors at 1k/10k/100k to find where the ANN index starts
             earning its keep. Single-match corpora are too small to show this, and
             quietly claiming the index matters at 15 clips would be dishonest.
  all        latency + retrieval.

Usage:
    python src/s05_benchmark.py --init-eval        # write a judgement template
    python src/s05_benchmark.py --mode all
    python src/s05_benchmark.py --mode scale --scale-sizes 1000 10000 100000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml

from common import (
    REPO_ROOT, dir_bytes, human_mb, load_config, connect, setup_logging, write_json,
)

log = setup_logging("s05_benchmark")


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    def pct(p):
        k = (len(s) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)
    return {
        "n": len(s),
        "min": round(s[0], 2),
        "p50": round(pct(0.50), 2),
        "p95": round(pct(0.95), 2),
        "p99": round(pct(0.99), 2),
        "max": round(s[-1], 2),
        "mean": round(statistics.fmean(s), 2),
    }


def warm_latency(base: str, queries: list[str], repeats: int, limit: int) -> dict:
    requests.post(f"{base}/warmup", timeout=300)
    # A few discarded iterations so JIT, pool and page cache are all settled.
    for q in queries[:3]:
        requests.get(f"{base}/search", params={"q": q, "limit": limit}, timeout=60)

    totals, encodes, searches, e2e = [], [], [], []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            r = requests.get(f"{base}/search", params={"q": q, "limit": limit}, timeout=60)
            wall = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            t = r.json()["timings_ms"]
            e2e.append(wall)
            totals.append(t["total"])
            encodes.append(t["text_encode"])
            searches.append(t["vector_search"])

    structured = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        requests.get(f"{base}/events", params={"limit": limit}, timeout=60).raise_for_status()
        structured.append((time.perf_counter() - t0) * 1000)

    return {
        "client_end_to_end_ms": percentiles(e2e),
        "server_total_ms": percentiles(totals),
        "text_encode_ms": percentiles(encodes),
        "vector_search_ms": percentiles(searches),
        "structured_only_ms": percentiles(structured),
        "raw_e2e_ms": [round(v, 2) for v in e2e],
    }


def cold_start(cfg, port: int, query: str) -> dict:
    """Launch a brand-new server process and time its first query.

    The first query of half-time is always cold, so reporting only warm numbers
    would flatter the system in exactly the situation it was built for.
    """
    env = dict(os.environ)
    env["GAA_CONFIG"] = str(REPO_ROOT / "config.yaml")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "s04_api:app",
         "--app-dir", str(REPO_ROOT / "src"), "--port", str(port), "--log-level", "warning"],
        cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        t_boot = time.perf_counter()
        for _ in range(120):
            try:
                if requests.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("cold-start server never became healthy")
        boot_s = time.perf_counter() - t_boot

        t0 = time.perf_counter()
        r = requests.get(f"{base}/search", params={"q": query, "limit": 5}, timeout=300)
        first_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()

        t0 = time.perf_counter()
        requests.get(f"{base}/search", params={"q": query, "limit": 5}, timeout=60).raise_for_status()
        second_ms = (time.perf_counter() - t0) * 1000

        return {
            "process_boot_to_healthy_s": round(boot_s, 2),
            "first_query_ms": round(first_ms, 1),
            "second_query_ms": round(second_ms, 1),
            "model_load_cost_ms": round(first_ms - second_ms, 1),
        }
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            proc.terminate()
        proc.wait(timeout=20)


def playback_ttfb(base: str, clip_ids: list[int]) -> dict:
    """Time to first byte on a ranged request — what the analyst experiences as
    'the video started'."""
    times = []
    for cid in clip_ids:
        t0 = time.perf_counter()
        with requests.get(f"{base}/clip/{cid}", headers={"Range": "bytes=0-65535"},
                          stream=True, timeout=30) as r:
            r.raise_for_status()
            next(r.iter_content(chunk_size=4096))
            times.append((time.perf_counter() - t0) * 1000)
    return percentiles(times)


def storage_and_ingest(cfg) -> dict:
    out_dir = cfg.get_path("paths.outputs_dir")
    ingest = json.loads((out_dir / "ingest_stats.json").read_text()) if (out_dir / "ingest_stats.json").exists() else {}
    clips = json.loads((out_dir / "clip_stats.json").read_text()) if (out_dir / "clip_stats.json").exists() else {}
    embed = json.loads((out_dir / "embed_stats.json").read_text()) if (out_dir / "embed_stats.json").exists() else {}

    duration_s = (ingest.get("media") or {}).get("duration_s") or clips.get("source_duration_s") or 0
    minutes = max(duration_s / 60, 1e-6)

    proxy_mb = human_mb(dir_bytes(cfg.get_path("paths.proxy_dir")))
    clips_mb = human_mb(dir_bytes(cfg.get_path("paths.clips_dir")))
    thumbs_mb = human_mb(dir_bytes(cfg.get_path("paths.thumbs_dir")))
    total_mb = proxy_mb + clips_mb + thumbs_mb

    proxy_s = (ingest.get("proxy") or {}).get("build_seconds") or 0
    clip_s = clips.get("wall_clock_s") or 0
    embed_s = embed.get("embed_wall_clock_s") or 0
    pipeline_s = proxy_s + clip_s + embed_s

    return {
        "match_duration_s": round(duration_s, 1),
        "storage_mb": {"proxy": proxy_mb, "clips": clips_mb, "thumbs": thumbs_mb, "total": total_mb},
        "mb_per_match_minute": round(total_mb / minutes, 2),
        "gb_per_35min_half": round((total_mb / minutes) * 35 / 1024, 3),
        "pipeline_seconds": {
            "proxy": round(proxy_s, 1), "clip_extraction": round(clip_s, 1),
            "embedding": round(embed_s, 1), "total": round(pipeline_s, 1),
        },
        "ingest_realtime_factor": round(duration_s / pipeline_s, 2) if pipeline_s else None,
        "minutes_to_ingest_35min_half": round(35 / (duration_s / 60 / (pipeline_s / 60)), 1)
        if pipeline_s and duration_s else None,
    }


def latency_budget(warm: dict, ttfb: dict, budget_s: float) -> dict:
    """The whole human-factors argument in one table: of the seconds an analyst
    has, how many does the system take and how many are left for thinking?"""
    per_query_ms = warm["client_end_to_end_ms"].get("p95", 0)
    playback_ms = ttfb.get("p95", 0) if ttfb else 0
    # A realistic task is not one query: participants reformulated.
    assumed_queries = 3
    system_ms = per_query_ms * assumed_queries + playback_ms
    return {
        "budget_s": budget_s,
        "assumed_queries_per_task": assumed_queries,
        "system_ms_p95": round(system_ms, 1),
        "system_share_of_budget_pct": round(100 * (system_ms / 1000) / budget_s, 2),
        "seconds_left_for_the_analyst": round(budget_s - system_ms / 1000, 1),
    }


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------
def dcg(gains: list[float]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def score_ranking(ranked_ids: list[int], judgments: dict[int, int], k_values=(1, 3, 5)) -> dict:
    gains = [judgments.get(cid, 0) for cid in ranked_ids]
    n_relevant = sum(1 for v in judgments.values() if v > 0)

    out = {}
    for k in k_values:
        top = gains[:k]
        out[f"P@{k}"] = round(sum(1 for g in top if g > 0) / k, 4)
    top10 = gains[:10]
    out["R@10"] = round(sum(1 for g in top10 if g > 0) / n_relevant, 4) if n_relevant else None
    first = next((i + 1 for i, g in enumerate(gains) if g > 0), None)
    out["RR"] = round(1 / first, 4) if first else 0.0
    ideal = sorted(judgments.values(), reverse=True)[:10]
    idcg = dcg([float(g) for g in ideal])
    out["nDCG@10"] = round(dcg([float(g) for g in top10]) / idcg, 4) if idcg else None
    return out


def init_eval_template(cfg, path: Path) -> None:
    """Dump the clip inventory into a YAML template ready for hand-judging."""
    conn = connect(cfg)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.clip_id, e.event_type, e.t_peak_s, e.pitch_zone, e.confidence
            FROM clips c JOIN events e ON e.event_id = c.event_id ORDER BY e.t_peak_s
        """)
        clips = cur.fetchall()
    conn.close()

    inventory = [
        {"clip_id": c[0], "event_type": c[1], "t_peak_s": round(c[2] or 0, 1),
         "zone": c[3], "confidence": round(c[4], 3) if c[4] is not None else None}
        for c in clips
    ]
    seed_queries = [
        "a contested kickout with players jumping for the ball",
        "a kickout that went short to a defender",
        "a long kickout landing in the middle third",
        "players competing for a high ball near the sideline",
        "a kickout won cleanly and uncontested",
        "a breaking ball after a midfield contest",
        "a kickout under pressure from the press",
        "two players jumping together for possession",
    ]
    doc = {
        "_instructions": (
            "Judge each query against the clip inventory below. Grades: 2 = clearly "
            "relevant, 1 = partially relevant, 0 = not relevant (omit these). Judge "
            "before you look at any system output, otherwise you are measuring your "
            "own anchoring rather than the retrieval."
        ),
        "_inventory": inventory,
        "queries": [
            {"id": f"q{i+1:02d}", "text": t,
             "filters": {"event_type": "kickout_contest"},
             "relevant": {}}
            for i, t in enumerate(seed_queries)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True, width=100)
    log.info("Wrote judgement template to %s (%d clips listed).", path, len(inventory))
    log.info("Fill in `relevant:` for each query, add more queries to reach 15–20, then rerun.")


def run_retrieval_eval(base: str, queries_path: Path, limit: int = 10) -> dict:
    with open(queries_path) as fh:
        doc = yaml.safe_load(fh)
    queries = [q for q in doc.get("queries", []) if q.get("relevant")]
    skipped = len(doc.get("queries", [])) - len(queries)
    if not queries:
        raise SystemExit(
            f"No judged queries in {queries_path}. Fill in the `relevant:` maps first."
        )
    if skipped:
        log.warning("Skipping %d unjudged queries.", skipped)

    per_mode: dict[str, list[dict]] = {"semantic": [], "hybrid": []}
    per_query = []

    for q in queries:
        judgments = {int(k): int(v) for k, v in q["relevant"].items()}
        row = {"id": q.get("id"), "text": q["text"], "n_relevant": len(judgments)}
        for mode in ("semantic", "hybrid"):
            params = {"q": q["text"], "limit": limit, "mode": mode}
            if mode == "hybrid":
                params.update({k: v for k, v in (q.get("filters") or {}).items() if v is not None})
            r = requests.get(f"{base}/search", params=params, timeout=60)
            r.raise_for_status()
            ranked = [res["clip_id"] for res in r.json()["results"]]
            scores = score_ranking(ranked, judgments)
            per_mode[mode].append(scores)
            row[mode] = {"ranked": ranked, **scores}
        per_query.append(row)

    def aggregate(rows: list[dict]) -> dict:
        keys = ["P@1", "P@3", "P@5", "R@10", "nDCG@10"]
        agg = {k: round(statistics.fmean([r[k] for r in rows if r.get(k) is not None]), 4)
               for k in keys if any(r.get(k) is not None for r in rows)}
        agg["MRR"] = round(statistics.fmean([r["RR"] for r in rows]), 4)
        return agg

    semantic = aggregate(per_mode["semantic"])
    hybrid = aggregate(per_mode["hybrid"])
    delta = {k: round(hybrid[k] - semantic[k], 4) for k in hybrid if k in semantic}

    return {
        "n_queries_judged": len(queries),
        "semantic": semantic,
        "hybrid": hybrid,
        "delta_hybrid_minus_semantic": delta,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Synthetic scale
# ---------------------------------------------------------------------------
def scale_test(cfg, sizes: list[int], n_probes: int = 30) -> dict:
    """Insert random unit vectors into a scratch table and time ANN vs seq scan.

    A single match yields tens of clips. A season across a county setup yields tens
    of thousands. This measures which regime the index is actually for, rather than
    assuming it.
    """
    dim = int(cfg.dot("embed.dim", 512))
    m = int(cfg.dot("index.m", 16))
    efc = int(cfg.dot("index.ef_construction", 64))
    rng = np.random.default_rng(42)
    conn = connect(cfg)
    results = []

    try:
        for n in sizes:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS scale_bench")
                cur.execute(f"CREATE TABLE scale_bench (id SERIAL PRIMARY KEY, embedding VECTOR({dim}))")
            log.info("Populating %d synthetic vectors…", n)
            batch = 2000
            with conn.cursor() as cur:
                for start in range(0, n, batch):
                    count = min(batch, n - start)
                    vecs = rng.standard_normal((count, dim)).astype(np.float32)
                    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
                    cur.executemany("INSERT INTO scale_bench (embedding) VALUES (%s)",
                                    [(v,) for v in vecs])

            probes = rng.standard_normal((n_probes, dim)).astype(np.float32)
            probes /= np.linalg.norm(probes, axis=1, keepdims=True)

            def time_queries() -> list[float]:
                times = []
                with conn.cursor() as cur:
                    for p in probes:
                        t0 = time.perf_counter()
                        cur.execute("SELECT id FROM scale_bench ORDER BY embedding <=> %s LIMIT 5", (p,))
                        cur.fetchall()
                        times.append((time.perf_counter() - t0) * 1000)
                return times

            with conn.cursor() as cur:
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute("ANALYZE scale_bench")
            flat = time_queries()

            with conn.cursor() as cur:
                t0 = time.perf_counter()
                cur.execute(
                    f"CREATE INDEX scale_bench_hnsw ON scale_bench USING hnsw "
                    f"(embedding vector_cosine_ops) WITH (m={m}, ef_construction={efc})"
                )
                build_s = time.perf_counter() - t0
                cur.execute("ANALYZE scale_bench")
                cur.execute("SELECT pg_relation_size('scale_bench_hnsw'), pg_relation_size('scale_bench')")
                idx_bytes, tbl_bytes = cur.fetchone()
            ann = time_queries()

            results.append({
                "n_vectors": n,
                "exact_scan_ms": percentiles(flat),
                "hnsw_ms": percentiles(ann),
                "speedup_p50": round(percentiles(flat)["p50"] / max(percentiles(ann)["p50"], 1e-6), 2),
                "index_build_s": round(build_s, 2),
                "index_mb": human_mb(idx_bytes),
                "table_mb": human_mb(tbl_bytes),
            })
            log.info("  n=%-7d exact p50 %.2f ms | hnsw p50 %.2f ms | %.1fx",
                     n, results[-1]["exact_scan_ms"]["p50"],
                     results[-1]["hnsw_ms"]["p50"], results[-1]["speedup_p50"])
    finally:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS scale_bench")
        conn.close()

    return {"dim": dim, "params": {"m": m, "ef_construction": efc}, "results": results}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_latency(bench: dict, out_dir: Path, budget_s: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = bench["warm"]["raw_e2e_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(raw, bins=min(30, max(8, len(raw) // 4)), edgecolor="black", alpha=0.8)
    for p, style in [("p50", "-"), ("p95", "--"), ("p99", ":")]:
        v = bench["warm"]["client_end_to_end_ms"][p]
        axes[0].axvline(v, linestyle=style, color="black", label=f"{p} = {v:.0f} ms")
    axes[0].set_xlabel("end-to-end query latency (ms)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Warm query latency")
    axes[0].legend(fontsize=8)

    xs = np.sort(raw)
    axes[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs))
    axes[1].axvline(500, linestyle="--", color="black", label="500 ms target")
    axes[1].set_xlabel("latency (ms)")
    axes[1].set_ylabel("cumulative proportion")
    axes[1].set_title("Latency CDF")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "latency.png", dpi=150)
    plt.close(fig)

    budget = bench.get("budget")
    if budget:
        fig, ax = plt.subplots(figsize=(8, 2.2))
        used = budget["system_ms_p95"] / 1000
        ax.barh([0], [used], color="0.35", label=f"system: {used:.2f} s")
        ax.barh([0], [budget_s - used], left=[used], color="0.85",
                label=f"analyst: {budget_s - used:.1f} s")
        ax.set_xlim(0, budget_s)
        ax.set_yticks([])
        ax.set_xlabel("seconds of the task budget")
        ax.set_title(f"Where the {budget_s:.0f}-second budget goes (p95, 3 queries + playback)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "latency_budget.png", dpi=150)
        plt.close(fig)


def plot_retrieval(evals: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ["P@1", "P@3", "P@5", "R@10", "MRR", "nDCG@10"]
            if k in evals["semantic"] and k in evals["hybrid"]]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, [evals["semantic"][k] for k in keys], width=0.4,
           label="semantic only", color="0.7", edgecolor="black")
    ax.bar(x + 0.2, [evals["hybrid"][k] for k in keys], width=0.4,
           label="hybrid (SQL pre-filter + vector)", color="0.3", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title(f"Retrieval quality, n={evals['n_queries_judged']} judged queries")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "retrieval_quality.png", dpi=150)
    plt.close(fig)


def plot_scale(scale: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n_vectors"] for r in scale["results"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, [r["exact_scan_ms"]["p50"] for r in scale["results"]], "o-", label="exact scan")
    ax.plot(ns, [r["hnsw_ms"]["p50"] for r in scale["results"]], "s-", label="HNSW")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clips in corpus")
    ax.set_ylabel("median query latency (ms)")
    ax.set_title("Where the ANN index starts to matter")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "scale.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the retrieval layer.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["latency", "retrieval", "scale", "all"], default="all")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--queries", default="eval/queries.yaml")
    ap.add_argument("--init-eval", action="store_true", help="write a judgement template and exit")
    ap.add_argument("--skip-cold", action="store_true")
    ap.add_argument("--cold-port", type=int, default=8099)
    ap.add_argument("--scale-sizes", type=int, nargs="*", default=[1000, 10000, 100000])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get_path("paths.outputs_dir")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    queries_path = Path(args.queries)
    if not queries_path.is_absolute():
        queries_path = REPO_ROOT / queries_path

    if args.init_eval:
        init_eval_template(cfg, queries_path)
        return

    base = args.base_url or f"http://{cfg.dot('api.host','127.0.0.1')}:{cfg.dot('api.port',8000)}"
    budget_s = float(cfg.dot("pilot.task_budget_s", 90))

    if args.mode == "scale":
        scale = scale_test(cfg, args.scale_sizes)
        write_json(out_dir / "scale_bench.json", scale)
        plot_scale(scale, fig_dir)
        log.info("Wrote %s and figures/scale.png", out_dir / "scale_bench.json")
        return

    try:
        health = requests.get(f"{base}/health", timeout=10).json()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Cannot reach the API at {base} ({exc}).\n"
            "Start it first:  uvicorn s04_api:app --app-dir src --port 8000"
        )
    if not health.get("indexed_clips"):
        raise SystemExit("The API reports zero indexed clips. Run s01–s03 first.")
    log.info("API healthy: %s clips indexed.", health["indexed_clips"])

    if args.mode in ("latency", "all"):
        probe_queries = [
            "a contested kickout", "players jumping for a high ball",
            "a short kickout to a defender", "a long ball into the middle third",
            "a kickout won cleanly",
        ]
        log.info("Warm latency: %d queries × %d repeats…", len(probe_queries), args.repeats)
        warm = warm_latency(base, probe_queries, args.repeats, args.limit)

        clip_ids = [r["clip_id"] for r in
                    requests.get(f"{base}/events", params={"limit": 5}, timeout=30).json()["results"]]
        ttfb = playback_ttfb(base, clip_ids) if clip_ids else {}

        cold = {} if args.skip_cold else cold_start(cfg, args.cold_port, "a contested kickout")
        if cold:
            log.info("Cold start: first query %.0f ms (model load ≈ %.0f ms of it).",
                     cold["first_query_ms"], cold["model_load_cost_ms"])

        bench = {
            "base_url": base,
            "health": health,
            "warm": warm,
            "cold": cold,
            "playback_ttfb_ms": ttfb,
            "resources": storage_and_ingest(cfg),
        }
        bench["budget"] = latency_budget(warm, ttfb, budget_s)
        write_json(out_dir / "benchmark.json", bench)
        plot_latency(bench, fig_dir, budget_s)

        e2e = warm["client_end_to_end_ms"]
        log.info("Warm p50 %.0f ms | p95 %.0f ms | p99 %.0f ms", e2e["p50"], e2e["p95"], e2e["p99"])
        log.info("Text encode is %.0f%% of server time (p50).",
                 100 * warm["text_encode_ms"]["p50"] / max(warm["server_total_ms"]["p50"], 1e-6))
        log.info("Budget: system takes %.2fs of %.0fs, leaving %.1fs to think.",
                 bench["budget"]["system_ms_p95"] / 1000, budget_s,
                 bench["budget"]["seconds_left_for_the_analyst"])
        log.info("Wrote %s", out_dir / "benchmark.json")

    if args.mode in ("retrieval", "all"):
        if not queries_path.exists():
            log.warning(
                "No query set at %s — skipping retrieval evaluation. "
                "Create one with: python src/s05_benchmark.py --init-eval", queries_path,
            )
        else:
            evals = run_retrieval_eval(base, queries_path, limit=10)
            write_json(out_dir / "retrieval_eval.json", evals)
            plot_retrieval(evals, fig_dir)
            log.info("Semantic: %s", evals["semantic"])
            log.info("Hybrid:   %s", evals["hybrid"])
            log.info("Delta:    %s", evals["delta_hybrid_minus_semantic"])
            log.info("Wrote %s", out_dir / "retrieval_eval.json")


if __name__ == "__main__":
    main()
