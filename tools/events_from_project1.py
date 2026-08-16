#!/usr/bin/env python3
"""Build an events CSV for data_2.mp4 from Project 1's artefacts.

WHY NOT s07's events_pred.csv
-----------------------------
Project 1 can emit predicted events, and for lgf26_final_w1 it does. It is the
wrong input here, for a reason worth stating plainly rather than working around
quietly: Project 1's own evaluation puts that detector's precision@8 at 1/8, and
its `leap` term was found to be scoring contests backwards (docs/10
`framing_scale`). Indexing those predictions would fold detector error into
Project 2's retrieval numbers, and a retrieval evaluation that cannot separate
"the ranker failed" from "the clip was not a kickout" measures nothing.

Project 1 also produced HAND-MARKED kickout timestamps for data_2 - 19 of them,
one skeleton per extracted segment - and those are the better input. They are
still a Project 1 artefact: the segmentation that produced them
(tools/extract_segments.py) and the timestamp collection are both Project 1
work. Using them means Project 2's retrieval metrics measure retrieval.

TIME BASE
---------
Two clocks exist and confusing them silently shifts every clip:

  segment-local  t_peak_s_approx   0 at the start of each extracted segment
  compilation    source_s          0 at the start of data_2.mp4

The skeletons carry BOTH. `source_s` is used, and the segment map is used to
re-derive it independently as a check, so a mistake in either fails loudly here
rather than showing up as clips that open two seconds late.

Usage:
    python tools/events_from_project1.py
    python tools/events_from_project1.py --check-only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_config, setup_logging  # noqa: E402

log = setup_logging("events_from_p1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project1", default="../gaa-kickout-vision",
                    help="path to the gaa-kickout-vision repo")
    ap.add_argument("--source-id", default="data_2")
    ap.add_argument("--out", default="data/events_data_2.csv")
    ap.add_argument("--pad-before", type=float, default=None,
                    help="defaults to clips.pad_before_s in config.yaml")
    ap.add_argument("--pad-after", type=float, default=None)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    p1 = (ROOT / args.project1).resolve()
    if not p1.exists():
        raise SystemExit(f"Project 1 not found at {p1}")

    pad_b = args.pad_before if args.pad_before is not None else float(cfg.dot("clips.pad_before_s", 3.0))
    pad_a = args.pad_after if args.pad_after is not None else float(cfg.dot("clips.pad_after_s", 5.0))

    smap_path = p1 / "data" / "gt" / f"{args.source_id}_segment_map.csv"
    if not smap_path.exists():
        raise SystemExit(f"segment map not found: {smap_path}")
    smap = pd.read_csv(smap_path)
    log.info("segment map: %d segments of %s", len(smap), args.source_id)

    rows, mismatches = [], []
    for seg in smap.itertuples():
        sk = p1 / "data" / "gt" / seg.video_id / "gt_events_skeleton.csv"
        if not sk.exists():
            log.warning("  %s: no skeleton, skipped", seg.video_id)
            continue
        df = pd.read_csv(sk)
        for r in df.itertuples():
            # authoritative compilation time
            t_src = float(r.source_s)
            # independent re-derivation from the segment map; these must agree
            t_derived = float(seg.source_start_s) + float(r.t_peak_s_approx)
            if abs(t_src - t_derived) > 0.5:
                mismatches.append(
                    f"{seg.video_id}/{r.event_id}: source_s={t_src:.1f} but "
                    f"segment_start({seg.source_start_s:.1f}) + local"
                    f"({r.t_peak_s_approx:.1f}) = {t_derived:.1f}")
            rows.append(dict(
                # Segment id MUST be in here. Every segment's skeleton restarts
                # its event_id at ko_001, so "data_2_ko_001" collides across
                # segments; s01 de-duplicates on event_id and silently kept 5 of
                # 19 events the first time this ran.
                event_id=f"{seg.video_id}_{r.event_id}",
                segment_id=seg.video_id,
                t_peak_s=round(t_src, 3),
                t_start_s=round(max(t_src - pad_b, 0.0), 3),
                t_end_s=round(t_src + pad_a, 3),
                event_type="kickout",
                confidence=1.0,          # hand-marked, not predicted
                source="project1_hand_marked",
                # Deliberately absent rather than guessed:
                #   pitch_zone  needs s04 registration, which needs hand-clicked
                #               landmarks and has never been run. A wrong zone is
                #               worse than a missing one - it is a PRE-FILTER, so
                #               a mislabelled clip becomes permanently unreachable.
                #   n_players   needs tracks.parquet for data_2; s02/s03 have only
                #               been run on lgf26_final_w1.
                pitch_zone=None,
                n_players_in_contest=None,
            ))

    if mismatches:
        for m in mismatches:
            log.error("  TIMEBASE MISMATCH %s", m)
        raise SystemExit(f"{len(mismatches)} timestamp(s) disagree between the "
                         "skeleton and the segment map. Fix before extracting "
                         "clips — every clip would be cut in the wrong place.")

    ev = pd.DataFrame(rows).sort_values("t_peak_s").reset_index(drop=True)

    # s01 de-duplicates on event_id, so a collision here does not raise — it
    # silently shrinks the corpus. Fail loudly instead.
    dupes = ev.event_id[ev.event_id.duplicated()].tolist()
    if dupes:
        raise SystemExit(f"duplicate event_id(s): {dupes}. Downstream de-duplication "
                         "would drop these without warning.")
    expected = int(smap.n_kickouts.sum())
    if len(ev) != expected:
        raise SystemExit(f"built {len(ev)} events but the segment map declares "
                         f"{expected} kickouts across its segments")

    log.info("%d kickouts, all timebase checks passed, ids unique, count matches "
             "the segment map", len(ev))
    log.info("  span %.1fs to %.1fs of the compilation", ev.t_peak_s.min(), ev.t_peak_s.max())
    log.info("  padding: -%.1fs / +%.1fs -> %.1fs per clip", pad_b, pad_a, pad_b + pad_a)

    # Overlapping windows would produce clips containing two kickouts, which
    # breaks the one-relevant-clip assumption the retrieval evaluation rests on.
    ev["gap_to_next_s"] = ev.t_peak_s.shift(-1) - ev.t_peak_s
    overlap = ev[ev.gap_to_next_s < (pad_b + pad_a)]
    if len(overlap):
        log.warning("  %d clip pair(s) overlap in time (gap < %.1fs): %s",
                    len(overlap), pad_b + pad_a,
                    ", ".join(f"{r.event_id}(+{r.gap_to_next_s:.1f}s)" for r in overlap.itertuples()))
        log.warning("  a query whose target is one of these can be satisfied by "
                    "either clip; the relevance judgements must say so")

    if args.check_only:
        log.info("check-only — nothing written")
        return

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.drop(columns=["gap_to_next_s"]).to_csv(out, index=False)

    prov = out.with_suffix(".provenance.json")
    prov.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_id": args.source_id,
        "n_events": int(len(ev)),
        "derived_from": {
            "segment_map": str(smap_path.relative_to(p1)),
            "skeletons": [f"data/gt/{s}/gt_events_skeleton.csv" for s in smap.video_id],
        },
        "event_source": "hand-marked kickout timestamps collected in Project 1",
        "why_not_s07": "Project 1's rule detector measured precision@8 = 1/8 and "
                       "had an inverted velocity term (docs/10 framing_scale). "
                       "Indexing its predictions would confound Project 2's "
                       "retrieval metrics with Project 1's detection error.",
        "padding_s": {"before": pad_b, "after": pad_a},
        "null_columns": {
            "pitch_zone": "s04 pitch registration never run (needs hand-clicked landmarks)",
            "n_players_in_contest": "no tracks.parquet for data_2",
        },
        "timebase": "compilation seconds, 0 = start of data_2.mp4; verified "
                    "against segment_start + segment-local time for every event",
    }, indent=2))
    log.info("wrote %s and %s", out.relative_to(ROOT), prov.name)


if __name__ == "__main__":
    main()
