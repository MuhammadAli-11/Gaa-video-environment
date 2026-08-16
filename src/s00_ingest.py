"""s00_ingest.py — register a match asset and build its scrubbing proxy.

Three jobs:
  1. ffprobe the source, write an `assets` row.
  2. Generate a low-resolution proxy with dense keyframes, so the UI can seek
     without decoding the full-rate stream. Standard practice in media asset
     management, and the reason scrubbing feels instant later.
  3. Verify that our view of the timeline matches Project 1's. If the detector
     counted a different number of frames, every timestamp downstream is wrong,
     and it is much cheaper to find that out here than during a pilot session.

Usage:
    python src/s00_ingest.py
    python src/s00_ingest.py --video data/raw/other_half.mp4 --force
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (
    Config, ensure_dirs, ffprobe, human_mb, load_config, connect,
    require_binaries, run, setup_logging, write_json,
)

log = setup_logging("s00_ingest")


def build_proxy(cfg: Config, src: Path, dest: Path) -> float:
    """Transcode a low-res proxy. Returns wall-clock seconds."""
    p = cfg["proxy"]
    ensure_dirs(dest.parent)
    t0 = time.perf_counter()
    run([
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-i", str(src),
        "-vf", f"scale=-2:{int(p['height'])}",
        "-c:v", "libx264", "-preset", str(p["preset"]), "-crf", str(p["crf"]),
        "-g", str(p["keyint"]), "-keyint_min", str(p["keyint"]),
        "-sc_threshold", "0",
        "-an",                      # analysts do not need audio to find a kickout
        "-movflags", "+faststart",  # moov atom first => playback starts immediately
        str(dest),
    ])
    return time.perf_counter() - t0


def check_alignment(cfg: Config, info) -> dict:
    """Compare our probe against Project 1's manifest, if one exists.

    Returns a dict with the measured offset. A non-zero offset does not stop the
    ingest, but it is recorded on the asset row and printed loudly.
    """
    result = {"manifest_found": False, "offset_s": 0.0, "warnings": []}
    try:
        manifest_path = cfg.opt_path("paths.manifest")
    except KeyError:
        return result
    if manifest_path is None:
        log.info("paths.manifest is null — timecode cross-check skipped. For the "
                 "data_2 corpus the events are in raw-compilation time by "
                 "construction and are already verified against Project 1's "
                 "segment map in tools/events_from_project1.py, which is the "
                 "stronger check of the two.")
        return result
    if not manifest_path.exists():
        log.warning("No Project 1 manifest at %s — skipping timecode check.", manifest_path)
        return result

    with open(manifest_path) as fh:
        man = json.load(fh)
    result["manifest_found"] = True

    m_dur = man.get("duration_s") or man.get("duration")
    m_fps = man.get("fps")
    m_frames = man.get("n_frames") or man.get("frame_count")

    if m_fps and info.fps and abs(float(m_fps) - info.fps) > 0.01:
        result["warnings"].append(
            f"fps mismatch: manifest={m_fps} probe={info.fps:.4f}. "
            "Event timestamps derived from frame indices will drift."
        )
    if m_frames and info.n_frames and int(m_frames) != int(info.n_frames):
        drift_frames = int(info.n_frames) - int(m_frames)
        drift_s = drift_frames / (info.fps or 25.0)
        result["offset_s"] = round(drift_s, 4)
        result["warnings"].append(
            f"frame count mismatch: manifest={m_frames} probe={info.n_frames} "
            f"({drift_frames} frames ≈ {drift_s:.3f}s)"
        )
    if m_dur and info.duration_s and abs(float(m_dur) - info.duration_s) > 0.1:
        result["warnings"].append(
            f"duration mismatch: manifest={float(m_dur):.3f}s probe={info.duration_s:.3f}s"
        )

    for w in result["warnings"]:
        log.warning("TIMECODE: %s", w)
    if not result["warnings"]:
        log.info("Timecode alignment against Project 1 manifest: OK.")
    return result


def upsert_asset(conn, info, proxy_path: Path, offset_s: float) -> int:
    row = info.as_row()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO assets (filename, duration_s, fps, n_frames, width, height,
                                codec, proxy_path, bytes, align_offset_s)
            VALUES (%(filename)s, %(duration_s)s, %(fps)s, %(n_frames)s, %(width)s,
                    %(height)s, %(codec)s, %(proxy_path)s, %(bytes)s, %(align_offset_s)s)
            ON CONFLICT (filename) DO UPDATE SET
                duration_s = EXCLUDED.duration_s,
                fps        = EXCLUDED.fps,
                n_frames   = EXCLUDED.n_frames,
                width      = EXCLUDED.width,
                height     = EXCLUDED.height,
                codec      = EXCLUDED.codec,
                proxy_path = EXCLUDED.proxy_path,
                bytes      = EXCLUDED.bytes,
                align_offset_s = EXCLUDED.align_offset_s,
                ingested_at = now()
            RETURNING asset_id
            """,
            {**row, "proxy_path": str(proxy_path), "align_offset_s": offset_s},
        )
        return cur.fetchone()[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a match asset.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", default=None, help="override paths.source_video")
    ap.add_argument("--force", action="store_true", help="rebuild the proxy even if it exists")
    ap.add_argument("--skip-proxy", action="store_true")
    args = ap.parse_args()

    require_binaries("ffmpeg", "ffprobe")
    cfg = load_config(args.config)

    src = Path(args.video).resolve() if args.video else cfg.get_path("paths.source_video")
    if not src.exists():
        raise SystemExit(
            f"Source video not found: {src}\n"
            "Put the standardised working clip from Project 1 there, or pass --video."
        )

    log.info("Probing %s", src)
    info = ffprobe(src)
    log.info(
        "  %sx%s  %.3f fps  %.1fs  %s  %.1f MB",
        info.width, info.height, info.fps or 0, info.duration_s, info.codec,
        human_mb(info.bytes),
    )

    align = check_alignment(cfg, info)

    proxy_dir = cfg.get_path("paths.proxy_dir")
    proxy_path = proxy_dir / f"{src.stem}_proxy.mp4"
    proxy_seconds = 0.0
    if args.skip_proxy:
        log.info("Skipping proxy generation (--skip-proxy).")
    elif proxy_path.exists() and not args.force:
        log.info("Proxy already present at %s (use --force to rebuild).", proxy_path)
    else:
        log.info("Building %sp proxy…", cfg.dot("proxy.height"))
        proxy_seconds = build_proxy(cfg, src, proxy_path)
        log.info(
            "  proxy built in %.1fs (%.1fx realtime), %.1f MB",
            proxy_seconds, info.duration_s / max(proxy_seconds, 1e-6),
            human_mb(proxy_path.stat().st_size),
        )

    conn = connect(cfg)
    asset_id = upsert_asset(conn, info, proxy_path, align["offset_s"])
    conn.close()
    log.info("assets.asset_id = %s", asset_id)

    stats = {
        "asset_id": asset_id,
        "source": str(src),
        "media": info.as_row(),
        "proxy": {
            "path": str(proxy_path),
            "build_seconds": round(proxy_seconds, 3),
            "realtime_factor": round(info.duration_s / proxy_seconds, 2) if proxy_seconds else None,
            "mb": human_mb(proxy_path.stat().st_size) if proxy_path.exists() else None,
        },
        "alignment": align,
    }
    out = write_json(cfg.get_path("paths.outputs_dir") / "ingest_stats.json", stats)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
