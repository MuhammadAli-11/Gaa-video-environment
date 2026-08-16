"""make_demo_events.py — stand-in events so the retrieval layer can be built and
tested before Project 1's detector has finished running.

Writes an events_pred.csv with the same column contract the real detector emits.
Useful on the Friday night when the database, API and UI need to exist but the
model is still training. Delete the file and repoint config.paths.events_csv at
the real output as soon as you have one.

Usage:
    python tools/make_demo_events.py --n 15
    python tools/make_demo_events.py --video data/raw/match.mp4 --out data/demo_events.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ffprobe, load_config, setup_logging  # noqa: E402

log = setup_logging("make_demo_events")

ZONES = ["own_third", "middle", "opp_third"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="data/demo_events_pred.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config(args.config)
    video = Path(args.video).resolve() if args.video else cfg.get_path("paths.source_video")
    info = ffprobe(video)
    rng = np.random.default_rng(args.seed)

    # Spread events across the timeline with a margin at each end so the padded
    # clip windows stay inside the footage.
    margin = 8.0
    peaks = np.sort(rng.uniform(margin, max(info.duration_s - margin, margin + 1), args.n))

    df = pd.DataFrame({
        "event_id": [f"demo_{i:03d}" for i in range(args.n)],
        "event_type": "kickout_contest",
        "t_start_s": np.round(peaks - 2.0, 2),
        "t_end_s": np.round(peaks + 3.0, 2),
        "t_peak_s": np.round(peaks, 2),
        "confidence": np.round(rng.uniform(0.45, 0.98, args.n), 3),
        "n_players": rng.integers(2, 9, args.n),
        "pitch_zone": rng.choice(ZONES, args.n),
    })

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    log.info("Wrote %d stand-in events to %s", args.n, out)
    log.info("Point config.paths.events_csv here, or run:")
    log.info("    python src/s01_extract_clips.py --events %s", out)
    log.warning("These are synthetic timestamps. They are for plumbing, not for evaluation.")


if __name__ == "__main__":
    main()
