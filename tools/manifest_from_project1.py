"""manifest_from_project1.py — bridge Project 1's frame manifest into the JSON
summary s00_ingest expects for its timecode check.

Project 1 writes data/interim/<video_id>/manifest.csv: one row per frame, the
authoritative statement of how many frames it believes exist and what timestamp
each carries. s00_ingest.check_alignment wants a small JSON summary instead.

The values here are read from Project 1's artifacts — the manifest row count and
the fps recorded in provenance_s01.json. They are deliberately NOT taken from an
ffprobe of the clip. Probing here would compare Project 2's probe against Project
2's probe and pass unconditionally, which is worse than having no check at all:
it would look like verification while testing nothing.

Usage:
    python tools/manifest_from_project1.py --video-id lgf26_final_w1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT1 = REPO_ROOT.parent / "gaa-kickout-vision"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--project1", default=str(PROJECT1))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    interim = Path(args.project1).resolve() / "data" / "interim" / args.video_id
    manifest_csv = interim / "manifest.csv"
    provenance = interim / "provenance_s01.json"

    if not manifest_csv.exists():
        raise SystemExit(f"no Project 1 manifest at {manifest_csv} — run its s01 first")

    man = pd.read_csv(manifest_csv)
    n_frames = len(man)

    fps = None
    trim = {}
    if provenance.exists():
        prov = json.loads(provenance.read_text())
        params = prov.get("params", {})
        fps = params.get("fps")
        trim = {
            "segment_start": params.get("segment_start"),
            "segment_duration": params.get("segment_duration"),
            "run_id": prov.get("run_id"),
            "git_sha": prov.get("git_sha"),
        }
    if not fps:
        # Fall back to the manifest's own spacing rather than guessing 25.
        step = man.timestamp_s.diff().dropna().median()
        fps = round(1.0 / step, 4) if step else None
    if not fps:
        raise SystemExit("could not determine fps from Project 1 artifacts")

    summary = {
        "video_id": args.video_id,
        "n_frames": int(n_frames),
        "fps": float(fps),
        "duration_s": round(n_frames / float(fps), 6),
        "first_timestamp_s": float(man.timestamp_s.iloc[0]),
        "last_timestamp_s": float(man.timestamp_s.iloc[-1]),
        "source": str(manifest_csv),
        "derived_from": "project1_manifest_and_provenance",
        **trim,
    }

    out = Path(args.out) if args.out else REPO_ROOT / "data" / f"{args.video_id}_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")
    for k in ("n_frames", "fps", "duration_s", "segment_start", "run_id"):
        print(f"  {k:16} {summary.get(k)}")


if __name__ == "__main__":
    main()
