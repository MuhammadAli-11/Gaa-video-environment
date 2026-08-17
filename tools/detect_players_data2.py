#!/usr/bin/env python3
"""Populate n_players_in_contest for the data_2 clips using Project 1's detector.

Calls gaa-kickout-vision's s02_detect.py and s03_track.py by subprocess. It never
reimplements them: the whole point is that this number comes from the same
detector and tracker whose failure modes Project 1 has measured.

WHY NOT COUNT TRACK IDS
-----------------------
Project 1 measured ByteTrack issuing 14-20 track identities per REAL player on
lgf26_final_w1 (33 GT identities, IDF1 0.103, no player tracked through 80% of
its lifetime). A distinct-track-ID count is therefore not a player count, it is a
fragmentation count. Both are computed here and reported side by side, because
the GAP between them is the result this project is after.

THE DE-DUPLICATED ESTIMATE
--------------------------
`median simultaneous detections per frame` over the contest window.

Chosen over spatial clustering of track centroids for three reasons:
  1. It never touches track identity, so it cannot inherit the fragmentation.
     Clustering centroids would still be clustering the output of a tracker that
     splits one player into 15 tracks moving together.
  2. The median over frames is robust to the transient spikes that a detector
     produces on a contest - a frame where two overlapping players yield three
     boxes does not move it.
  3. It has a single failure mode that is easy to state: it inherits DETECTION
     recall, measured by Project 1 at 0.294 @IoU 0.5 / 0.463 @IoU 0.3, so it
     UNDER-counts. It is a floor on the true player count, not an estimate of it.

Mean simultaneous detections is also reported, so the reader can see how much the
choice of central tendency matters.

Usage:
    python tools/detect_players_data2.py --limit 2      # smoke test
    python tools/detect_players_data2.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_config, setup_logging, run as run_cmd, FFMPEG  # noqa: E402

log = setup_logging("detect_data2")

P1 = (ROOT / ".." / "gaa-kickout-vision").resolve()
P1_CONFIG = "config_data2.yaml"


def p1_run_dir() -> Path:
    sys.path.insert(0, str(P1 / "src"))
    from lib.config import load_config as p1_load, run_id  # noqa: E402
    cfg = p1_load(P1 / P1_CONFIG)
    return P1 / cfg["paths"]["outputs"] / run_id(cfg)


def cut_working_clip(src: Path, dest: Path, t_start: float, dur: float, cfg) -> None:
    """Cut one clip to Project 1's working-clip spec: CFR 25 fps, 1280x720, yuv420p.

    Matches s01_prepare_footage.py's encode so the detector sees exactly what it
    would have seen had s01 produced this clip.
    """
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        FFMPEG, "-nostdin", "-v", "error", "-y",
        "-ss", f"{t_start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
        "-vf", "scale=1280:720", "-r", "25",
        "-fps_mode", "cfr", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an", str(dest),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips")
    ap.add_argument("--pad", type=float, default=None,
                    help="seconds either side of t_peak; defaults to the clip padding")
    ap.add_argument("--out", default="outputs/player_counts_data_2.csv")
    args = ap.parse_args()

    cfg = load_config()
    events = pd.read_csv(ROOT / cfg.dot("paths.events_csv"))
    if args.limit:
        events = events.head(args.limit)
    src = (ROOT / cfg.dot("paths.source_video")).resolve()
    pad_b = float(cfg.dot("clips.pad_before_s", 3.0))
    pad_a = float(cfg.dot("clips.pad_after_s", 5.0))

    clips_dir = P1 / "data" / "clips"
    run_dir = p1_run_dir()
    store = ROOT / "outputs" / "data_2_detect"
    store.mkdir(parents=True, exist_ok=True)
    log.info("Project 1 at %s (config %s)", P1, P1_CONFIG)
    log.info("Project 1 run dir: %s", run_dir.name)
    log.info("%d clips, source %s", len(events), src.name)

    rows = []
    t_all = time.time()
    for i, ev in enumerate(events.itertuples(), 1):
        vid = f"data_2_{ev.event_id.replace('data_2_', '')}"
        clip = clips_dir / f"{vid}_working.mp4"
        dur = float(ev.t_end_s) - float(ev.t_start_s)
        cut_working_clip(src, clip, float(ev.t_start_s), dur, cfg)

        dest = store / vid
        dest.mkdir(parents=True, exist_ok=True)
        det_f, trk_f = dest / "detections.parquet", dest / "tracks.parquet"

        # Detection is ~6 min a clip and tracking is seconds, so each stage is
        # checkpointed separately. A tracker failure must never cost the
        # detection pass that preceded it.
        t0 = time.time()
        did = []
        for stage, out_name, target in (("s02_detect.py", "detections.parquet", det_f),
                                        ("s03_track.py", "tracks.parquet", trk_f)):
            if target.exists():
                continue
            # s03 reads detections.parquet from the run dir, so put it back first
            if stage == "s03_track.py" and det_f.exists() and not (run_dir / "detections.parquet").exists():
                shutil.copy(str(det_f), str(run_dir / "detections.parquet"))
            r = subprocess.run(
                [sys.executable, str(P1 / "src" / stage),
                 "--config", P1_CONFIG, "--video-id", vid],
                cwd=P1, capture_output=True, text=True)
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "")[-1500:]
                log.error("%s failed for %s:\n%s", stage, vid, tail)
                raise SystemExit(1)
            # s02/s03 write flat into the run dir keyed only by run_id, so each
            # clip would overwrite the last. Move out immediately.
            shutil.move(str(run_dir / out_name), str(target))
            did.append(stage[:3])
        # leave nothing behind that the next clip's stage could mistake for its own
        for stale in ("detections.parquet", "tracks.parquet"):
            if (run_dir / stale).exists():
                (run_dir / stale).unlink()
        log.info("[%2d/%2d] %-22s %s %.1fs", i, len(events), vid,
                 "+".join(did) if did else "cached", time.time() - t0)

        det = pd.read_parquet(det_f)
        trk = pd.read_parquet(trk_f)

        # Contest window, centred on t_peak in CLIP-LOCAL time.
        pad = args.pad if args.pad is not None else min(pad_b, pad_a)
        t_local = float(ev.t_peak_s) - float(ev.t_start_s)
        lo, hi = t_local - pad, t_local + pad
        d = det[(det.timestamp_s >= lo) & (det.timestamp_s <= hi)]
        t = trk[(trk.timestamp_s >= lo) & (trk.timestamp_s <= hi)]

        per_frame = d.groupby("frame_idx").size()
        naive = int(t.track_id.nunique())
        dedup = int(round(per_frame.median())) if len(per_frame) else 0

        rows.append(dict(
            event_id=ev.event_id,
            t_peak_s=float(ev.t_peak_s),
            window_s=round(2 * pad, 1),
            n_frames=int(len(per_frame)),
            n_detections=int(len(d)),
            naive_n_players_track_ids=naive,
            dedup_n_players_median=dedup,
            dedup_n_players_mean=round(float(per_frame.mean()), 2) if len(per_frame) else 0.0,
            per_frame_p25=int(per_frame.quantile(.25)) if len(per_frame) else 0,
            per_frame_p75=int(per_frame.quantile(.75)) if len(per_frame) else 0,
            per_frame_max=int(per_frame.max()) if len(per_frame) else 0,
            inflation_factor=round(naive / dedup, 2) if dedup else np.nan,
            mean_conf=round(float(d.conf.mean()), 3) if len(d) else np.nan,
            median_bbox_h=round(float((d.y2 - d.y1).median()), 1) if len(d) else np.nan,
        ))

    out = pd.DataFrame(rows)
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    log.info("=" * 78)
    log.info("%d clips in %.1f min", len(out), (time.time() - t_all) / 60)
    log.info("naive track-ID count : median %.0f  range %d-%d",
             out.naive_n_players_track_ids.median(),
             out.naive_n_players_track_ids.min(), out.naive_n_players_track_ids.max())
    log.info("de-duplicated count  : median %.0f  range %d-%d",
             out.dedup_n_players_median.median(),
             out.dedup_n_players_median.min(), out.dedup_n_players_median.max())
    log.info("inflation factor     : median %.1fx range %.1f-%.1fx",
             out.inflation_factor.median(), out.inflation_factor.min(),
             out.inflation_factor.max())
    log.info("wrote %s", dest.relative_to(ROOT))

    (dest.with_suffix(".provenance.json")).write_text(json.dumps({
        "produced_by": "tools/detect_players_data2.py",
        "project1_config": P1_CONFIG,
        "project1_run_id": run_dir.name,
        "detector": "yolov8m.pt, imgsz 960, conf 0.25, class 0",
        "imgsz_deviation": "Project 1's config.yaml uses imgsz 1280. data_2 is "
                           "natively 854x478, so 1280 is upsampling. 960 agreed "
                           "with 1280 on 100% of reference boxes at IoU 0.5 (n=6 "
                           "frames) and ran 2.2x faster. NOT valid for lgf26_final.",
        "tracker": "bytetrack, per Project 1 config",
        "contest_window_s": 2 * (args.pad if args.pad is not None else min(pad_b, pad_a)),
        "dedup_method": "median simultaneous detections per frame over the contest window",
        "dedup_caveat": "inherits detection recall (Project 1: 0.294 @IoU0.5, "
                        "0.463 @IoU0.3), so this UNDER-counts. It is a floor on "
                        "the true player count, not an unbiased estimate.",
        "naive_caveat": "distinct track ids. Project 1 measured 14-20 identities "
                        "per real player, so this is a fragmentation count.",
        "n_clips": int(len(out)),
    }, indent=2))


if __name__ == "__main__":
    main()
