"""s01_extract_clips.py — turn detected events into retrievable clips.

Reads Project 1's `events_pred.csv`, writes `events` and `clips` rows, and cuts
one padded clip plus one thumbnail per event.

Two decisions worth defending at interview:

  * Padding is asymmetric (-3s / +5s around t_peak_s). Analysts need the build-up
    to judge a contest, and the outcome resolves after it. A symmetric window
    around the peak looks tidy and shows you the wrong thing.

  * Default cut mode is re-encode, not stream copy. Stream copy is roughly an
    order of magnitude faster but snaps boundaries to the nearest keyframe, which
    on a 250-frame GOP can be ten seconds adrift. When a clip does not start where
    the timestamp says it does, analysts stop trusting the timestamps, and a tool
    an analyst does not trust is not a faster tool. Both modes are implemented and
    s05 reports the throughput cost of the choice.

Usage:
    python src/s01_extract_clips.py
    python src/s01_extract_clips.py --mode stream_copy --limit 5
    python src/s01_extract_clips.py --events path/to/events_pred.csv --force
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from common import (
    Config, ensure_dirs, ffprobe, human_mb, load_config, connect,
    require_binaries, run, setup_logging, write_json,
)

log = setup_logging("s01_extract_clips")

# Project 1 may name its columns slightly differently. Map generously, fail loudly.
COLUMN_ALIASES = {
    "event_id":   ["event_id", "id", "idx", "event_idx"],
    "event_type": ["event_type", "type", "label", "class", "event"],
    "t_start_s":  ["t_start_s", "t_start", "start_s", "start", "start_time"],
    "t_end_s":    ["t_end_s", "t_end", "end_s", "end", "end_time"],
    "t_peak_s":   ["t_peak_s", "t_peak", "peak_s", "peak", "t_event_s"],
    "confidence": ["confidence", "conf", "score", "prob"],
    # n_players_in_contest is what gaa-kickout-vision's events_pred schema emits.
    "n_players":  ["n_players", "n_players_in_contest", "num_players",
                   "player_count", "n_tracks"],
    "pitch_zone": ["pitch_zone", "zone", "third", "region"],
    # Provenance of the timestamp: 'model' (a detector proposed it) or anything
    # else (a human marked it). Defaults to 'model' when absent, which is the
    # conservative reading of an unlabelled events file.
    "source":     ["source", "event_source", "provenance"],
}


def normalise_events(df: pd.DataFrame, default_type: str) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)

    unmapped = []
    for canonical, aliases in COLUMN_ALIASES.items():
        source = next((lower[a] for a in aliases if a in lower), None)
        out[canonical] = df[source] if source else None
        if source is None:
            unmapped.append(canonical)

    # "Map generously, fail loudly" only held for the required columns; the
    # optional ones used to become NULL in silence. A NULL here is a retrieval
    # filter that quietly cannot filter, which is exactly the kind of thing that
    # is noticed after the evaluation rather than before it.
    if unmapped:
        log.warning("no source column found for %s — stored as NULL. Available: %s",
                    ", ".join(unmapped), ", ".join(df.columns))

    if out["t_start_s"].isna().all() or out["t_end_s"].isna().all():
        raise SystemExit(
            "events_pred.csv needs start and end times. Columns found: "
            + ", ".join(df.columns)
        )

    out["t_start_s"] = out["t_start_s"].astype(float)
    out["t_end_s"] = out["t_end_s"].astype(float)
    out["t_peak_s"] = out["t_peak_s"].astype(float).where(
        out["t_peak_s"].notna(), (out["t_start_s"] + out["t_end_s"]) / 2.0
    )
    out["event_type"] = out["event_type"].fillna(default_type)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["n_players"] = pd.to_numeric(out["n_players"], errors="coerce").astype("Int64")
    out["ext_event_id"] = out["event_id"].where(out["event_id"].notna(),
                                                pd.Series(out.index, index=out.index)).astype(str)
    return out.sort_values("t_start_s").reset_index(drop=True)


def derive_zones(events: pd.DataFrame, tracks_path: Path | None) -> pd.DataFrame:
    """Best-effort pitch zone from Project 1's tracks, if it carries pitch coords.

    Looks for a normalised longitudinal coordinate (0 = own end line, 1 = opposition
    end line) and takes the mean player position inside each event window. Silently
    leaves the column NULL if the tracks file has no usable geometry — a wrong zone
    label is worse than a missing one, because the filter is a pre-filter and a
    wrong label makes a clip permanently unreachable.
    """
    if events["pitch_zone"].notna().any():
        log.info("pitch_zone already present in events file — not deriving.")
        return events
    if tracks_path is None:
        log.info("paths.tracks_parquet is null — pitch_zone stays NULL. "
                 "The zone pre-filter is unavailable for this corpus.")
        return events
    if not tracks_path.exists():
        log.info("No tracks parquet at %s — pitch_zone left NULL.", tracks_path)
        return events

    try:
        tracks = pd.read_parquet(tracks_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read tracks parquet (%s) — pitch_zone left NULL.", exc)
        return events

    cols = {c.lower(): c for c in tracks.columns}
    x_col = next((cols[c] for c in ("pitch_x_norm", "x_norm", "px_norm", "x_pitch") if c in cols), None)
    t_col = next((cols[c] for c in ("t_s", "time_s", "timestamp_s", "t") if c in cols), None)
    if not x_col or not t_col:
        log.info(
            "tracks parquet has no normalised pitch coordinates (looked for x_norm/t_s) — "
            "pitch_zone left NULL. Columns: %s", ", ".join(tracks.columns)[:200],
        )
        return events

    zones = []
    for _, ev in events.iterrows():
        window = tracks[(tracks[t_col] >= ev.t_start_s) & (tracks[t_col] <= ev.t_end_s)]
        if window.empty:
            zones.append(None)
            continue
        mean_x = float(window[x_col].mean())
        zones.append("own_third" if mean_x < 1 / 3 else ("middle" if mean_x < 2 / 3 else "opp_third"))
    events["pitch_zone"] = zones
    log.info("Derived pitch_zone for %d/%d events.", sum(z is not None for z in zones), len(zones))
    return events


def cut_clip(cfg: Config, src: Path, dest: Path, start: float, duration: float, mode: str) -> None:
    c = cfg["clips"]
    base = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
    if mode == "stream_copy":
        cmd = base + [
            "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero", str(dest),
        ]
    else:
        # Coarse input seek then accurate output seek: fast and frame-accurate.
        cmd = base + [
            "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", str(c["preset"]), "-crf", str(c["crf"]),
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dest),
        ]
    run(cmd)


def make_thumb(cfg: Config, src: Path, dest: Path, t_s: float) -> None:
    run([
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-ss", f"{max(t_s, 0):.3f}", "-i", str(src), "-frames:v", "1",
        "-vf", f"scale={int(cfg.dot('clips.thumb_width', 480))}:-2",
        "-q:v", "4", str(dest),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Cut clips for detected events.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--events", default=None, help="override paths.events_csv")
    ap.add_argument("--video", default=None, help="override paths.source_video")
    ap.add_argument("--mode", choices=["reencode", "stream_copy"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N events")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--default-type", default="kickout_contest")
    ap.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    args = ap.parse_args()

    require_binaries("ffmpeg", "ffprobe")
    cfg = load_config(args.config)
    mode = args.mode or cfg.dot("clips.mode", "reencode")

    src = Path(args.video).resolve() if args.video else cfg.get_path("paths.source_video")
    events_csv = Path(args.events).resolve() if args.events else cfg.get_path("paths.events_csv")
    if not src.exists():
        raise SystemExit(f"Source video not found: {src}")
    if not events_csv.exists():
        raise SystemExit(
            f"Events file not found: {events_csv}\n"
            "Point paths.events_csv at Project 1's output, or generate a stand-in with:\n"
            "    python tools/make_demo_events.py"
        )

    info = ffprobe(src)
    raw = pd.read_csv(events_csv)
    events = normalise_events(raw, args.default_type)
    events = derive_zones(events, cfg.opt_path("paths.tracks_parquet"))

    if args.min_confidence > 0:
        before = len(events)
        events = events[events["confidence"].fillna(1.0) >= args.min_confidence]
        log.info("Confidence filter kept %d/%d events.", len(events), before)
    if args.limit:
        events = events.head(args.limit)

    clips_dir = cfg.get_path("paths.clips_dir")
    thumbs_dir = cfg.get_path("paths.thumbs_dir")
    ensure_dirs(clips_dir, thumbs_dir)

    pad_b = float(cfg.dot("clips.pad_before_s", 3.0))
    pad_a = float(cfg.dot("clips.pad_after_s", 5.0))

    conn = connect(cfg)
    with conn.cursor() as cur:
        cur.execute("SELECT asset_id FROM assets WHERE filename = %s", (str(src),))
        row = cur.fetchone()
    if not row:
        raise SystemExit("Asset not ingested. Run: python src/s00_ingest.py")
    asset_id = row[0]

    log.info("Cutting %d clips from asset %s in '%s' mode…", len(events), asset_id, mode)
    t0 = time.perf_counter()
    n_written = 0

    for _, ev in events.iterrows():
        start = max(0.0, float(ev.t_peak_s) - pad_b)
        end = min(info.duration_s, float(ev.t_peak_s) + pad_a)
        duration = max(end - start, 0.5)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO events (asset_id, ext_event_id, event_type, t_start_s, t_end_s,
                                    t_peak_s, confidence, n_players, pitch_zone, source)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (asset_id, ext_event_id) DO UPDATE SET
                    event_type=EXCLUDED.event_type, t_start_s=EXCLUDED.t_start_s,
                    t_end_s=EXCLUDED.t_end_s, t_peak_s=EXCLUDED.t_peak_s,
                    confidence=EXCLUDED.confidence, n_players=EXCLUDED.n_players,
                    pitch_zone=EXCLUDED.pitch_zone, source=EXCLUDED.source
                RETURNING event_id
                """,
                (
                    asset_id, str(ev.ext_event_id), ev.event_type,
                    float(ev.t_start_s), float(ev.t_end_s), float(ev.t_peak_s),
                    None if pd.isna(ev.confidence) else float(ev.confidence),
                    None if pd.isna(ev.n_players) else int(ev.n_players),
                    ev.pitch_zone if isinstance(ev.pitch_zone, str) else None,
                    # Provenance of the TIMESTAMP, not of the clip. 'model' means a
                    # detector proposed it; anything else means a human marked it.
                    # The distinction decides whether a retrieval metric computed
                    # over these clips is confounded by detection error, so it is
                    # carried through rather than hardcoded.
                    getattr(ev, "source", None) if isinstance(getattr(ev, "source", None), str) else "model",
                ),
            )
            event_id = cur.fetchone()[0]

        clip_path = clips_dir / f"event_{event_id:04d}.mp4"
        thumb_path = thumbs_dir / f"event_{event_id:04d}.jpg"

        if clip_path.exists() and not args.force:
            log.debug("clip exists, skipping: %s", clip_path.name)
        else:
            cut_clip(cfg, src, clip_path, start, duration, mode)
            make_thumb(cfg, src, thumb_path, float(ev.t_peak_s))
            n_written += 1

        actual = ffprobe(clip_path)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO clips (event_id, path, thumb_path, duration_s, bytes)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (event_id) DO UPDATE SET
                    path=EXCLUDED.path, thumb_path=EXCLUDED.thumb_path,
                    duration_s=EXCLUDED.duration_s, bytes=EXCLUDED.bytes
                """,
                (event_id, str(clip_path), str(thumb_path), actual.duration_s, actual.bytes),
            )

    elapsed = time.perf_counter() - t0
    conn.close()

    clip_bytes = sum(f.stat().st_size for f in clips_dir.glob("*.mp4"))
    stats = {
        "asset_id": asset_id,
        "mode": mode,
        "n_events": int(len(events)),
        "n_clips_written": n_written,
        "padding": {"before_s": pad_b, "after_s": pad_a},
        "wall_clock_s": round(elapsed, 2),
        "seconds_per_clip": round(elapsed / max(n_written, 1), 3),
        "source_duration_s": round(info.duration_s, 2),
        "ingest_realtime_factor": round(info.duration_s / elapsed, 2) if elapsed else None,
        "clips_mb": human_mb(clip_bytes),
        "mb_per_match_minute": round(human_mb(clip_bytes) / max(info.duration_s / 60, 1e-6), 2),
    }
    log.info(
        "Done: %d clips in %.1fs (%.2f s/clip, %.1fx realtime), %.1f MB",
        n_written, elapsed, stats["seconds_per_clip"],
        stats["ingest_realtime_factor"] or 0, stats["clips_mb"],
    )
    out = write_json(cfg.get_path("paths.outputs_dir") / "clip_stats.json", stats)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
