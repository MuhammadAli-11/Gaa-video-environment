#!/usr/bin/env python3
"""How far does vision-layer identity fragmentation propagate into retrieval?

Three steps:
  1. Load the per-clip player counts, write the DE-DUPLICATED one into the DB.
     The naive track-ID count is NEVER written — it would corrupt the field.
  2. Quantify the gap: inflation factor per clip, and what a `n_players >= k`
     filter returns under each count.
  3. Re-run the six queries with and without the filter and report divergence.

The point is not that the naive count is a bit high. It is that the naive count
is a FRAGMENTATION count wearing the name of a player count, and a retrieval
filter cannot tell the difference.

Usage:
    python tools/analyse_metadata_propagation.py --write-db
    python tools/analyse_metadata_propagation.py --queries-only
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import connect, load_config, setup_logging  # noqa: E402

log = setup_logging("propagation")

QUERIES = [
    "contested kickout",
    "players jumping for a high ball",
    "goalkeeper restart",
    "long kick downfield",
    "players competing in the air",
    "short kickout to the side",
]


def api(path: str, **params) -> dict:
    url = "http://127.0.0.1:8000" + path + "?" + urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None})
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def rank_overlap(a: list, b: list) -> float:
    """Fraction of positions holding the same clip — order-sensitive."""
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="outputs/player_counts_data_2.csv")
    ap.add_argument("--write-db", action="store_true")
    ap.add_argument("--queries-only", action="store_true")
    ap.add_argument("--k", type=int, default=4, help="threshold for the n_players filter")
    args = ap.parse_args()

    cfg = load_config()
    counts = pd.read_csv(ROOT / args.counts)
    n = len(counts)

    # ---- 1. write the de-duplicated count only -------------------------
    if args.write_db and not args.queries_only:
        conn = connect(cfg)
        with conn.cursor() as cur:
            for r in counts.itertuples():
                cur.execute(
                    "UPDATE events SET n_players = %s WHERE ext_event_id = %s",
                    (int(r.dedup_n_players_median), r.event_id))
        try:
            conn.commit()
        except Exception:
            pass
        log.info("wrote de-duplicated n_players for %d clips (naive count NOT written)", n)

    if not args.queries_only:
        # ---- 2. the propagation ----------------------------------------
        infl = counts.inflation_factor.dropna()
        print("\n" + "=" * 78)
        print(f"TASK 2 — NAIVE vs DE-DUPLICATED PLAYER COUNT   (n = {n} clips)")
        print("=" * 78)
        print(f"{'event_id':<26}{'naive':>7}{'dedup':>7}{'infl':>7}{'det/frm p25-p75':>18}{'bbox h':>8}")
        for r in counts.itertuples():
            print(f"{r.event_id:<26}{r.naive_n_players_track_ids:>7}"
                  f"{r.dedup_n_players_median:>7}{r.inflation_factor:>6.1f}x"
                  f"{f'{r.per_frame_p25}-{r.per_frame_p75}':>18}{r.median_bbox_h:>8.0f}")

        print(f"\n{'':<26}{'naive':>7}{'dedup':>7}{'infl':>7}")
        for lab, fn in (("median", np.median), ("min", np.min), ("max", np.max)):
            print(f"{lab:<26}{fn(counts.naive_n_players_track_ids):>7.0f}"
                  f"{fn(counts.dedup_n_players_median):>7.0f}{fn(infl):>6.1f}x")
        print(f"\ninflation factor: median {infl.median():.1f}x, "
              f"IQR {infl.quantile(.25):.1f}-{infl.quantile(.75):.1f}x, "
              f"range {infl.min():.1f}-{infl.max():.1f}x  (n={len(infl)})")

        # does inflation track clip content?
        print("\nwhat drives the inflation? (Spearman rho, n=%d)" % n)
        for col, label in (("dedup_n_players_median", "players present"),
                           ("median_bbox_h", "median player size px"),
                           ("mean_conf", "mean detection confidence"),
                           ("n_detections", "total detections in window")):
            if counts[col].notna().sum() > 2 and counts[col].nunique() > 1:
                rho = counts[[col, "inflation_factor"]].corr(method="spearman").iloc[0, 1]
                print(f"  {label:<28} rho = {rho:+.3f}")

        # ---- what the filter returns under each count -------------------
        k = args.k
        print(f"\nwhat a filter `n_players >= {k}` returns, {n} clips indexed:")
        for label, col in (("naive track-ID count", "naive_n_players_track_ids"),
                           ("de-duplicated count", "dedup_n_players_median")):
            passing = int((counts[col] >= k).sum())
            print(f"  {label:<24} {passing:>3}/{n} clips pass "
                  f"({100*passing/n:5.1f}%)")
        for k2 in (4, 8, 12, 16, 20):
            a = int((counts.naive_n_players_track_ids >= k2).sum())
            b = int((counts.dedup_n_players_median >= k2).sum())
            print(f"    n_players >= {k2:<3} naive {a:>3}/{n}   dedup {b:>3}/{n}")

    # ---- 3. hybrid vs semantic ----------------------------------------
    print("\n" + "=" * 78)
    print(f"TASK 3 — HYBRID vs SEMANTIC, WITH THE FILTER POPULATED  (n = {len(QUERIES)} queries)")
    print("=" * 78)
    try:
        h = api("/health")
    except Exception as e:  # noqa: BLE001
        log.error("API not reachable (%s). Start it, then rerun --queries-only.", e)
        return
    log.info("API: %d clips indexed", h.get("indexed_clips"))

    k = args.k
    print(f"{'query':<34}{'identical':>10}{'jaccard':>9}{'rank ovl':>10}{'filters':>26}")
    ident = 0
    jac, ro = [], []
    for q in QUERIES:
        sem = api("/search", q=q, limit=5, mode="semantic")
        hyb = api("/search", q=q, limit=5, mode="hybrid", min_players=k)
        a = [r["clip_id"] for r in sem["results"]]
        b = [r["clip_id"] for r in hyb["results"]]
        same = a == b
        ident += same
        jac.append(jaccard(a, b))
        ro.append(rank_overlap(a, b))
        print(f"{q:<34}{str(same):>10}{jac[-1]:>9.2f}{ro[-1]:>10.2f}"
              f"{str(hyb.get('filters_applied')):>26}")
    print(f"\nidentical rankings: {ident}/{len(QUERIES)}   "
          f"mean Jaccard {np.mean(jac):.3f}   mean positional overlap {np.mean(ro):.3f}")
    if ident == len(QUERIES):
        print("-> the filter changes nothing. One weak field is not enough to "
              "separate the arms on a single-event-class corpus.")
    else:
        print(f"-> the arms now diverge on {len(QUERIES)-ident}/{len(QUERIES)} queries.")


if __name__ == "__main__":
    main()
