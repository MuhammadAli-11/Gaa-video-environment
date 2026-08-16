"""s02_embed.py — one vector per clip. The only GPU step in the pipeline.

Samples N keyframes per clip, runs the CLIP image tower, L2-normalises each frame
embedding, mean-pools, then re-normalises. Writes to `clip_embeddings`.

Mean pooling over frames is a deliberate simplification: it throws away temporal
order, which for an 8-second kickout clip matters less than it would for a longer
sequence, and it keeps the query path to a single vector comparison. If retrieval
quality were the target rather than latency, this is the first thing to revisit —
but the whole argument of this project is that it is not the target.

Cost: ~15 clips x 8 frames = 120 images. Minutes, not hours.

Usage:
    python src/s02_embed.py
    python src/s02_embed.py --force --device cuda
"""
from __future__ import annotations

import argparse
import io
import time

import numpy as np
from PIL import Image

from common import (
    grab_frame_png, load_config, connect, pick_device, require_binaries,
    setup_logging, write_json,
)

log = setup_logging("s02_embed")


def load_encoder(model_name: str, pretrained: str, device: str):
    """Import torch lazily so the rest of the repo runs without it installed."""
    import open_clip
    import torch

    t0 = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    load_s = time.perf_counter() - t0
    log.info("Loaded %s/%s on %s in %.1fs", model_name, pretrained, device, load_s)
    return model, preprocess, torch, load_s


def keyframe_times(duration_s: float, n: int) -> list[float]:
    """Uniform samples with a small inset, so we never land on a black first or
    last frame from the encoder's GOP boundary."""
    if duration_s <= 0:
        return [0.0]
    lo, hi = 0.06 * duration_s, 0.94 * duration_s
    return list(np.linspace(lo, hi, num=max(n, 1)))


def embed_clip(path: str, duration_s: float, n_frames: int, model, preprocess, torch, device: str):
    frames = []
    for t in keyframe_times(duration_s, n_frames):
        try:
            frames.append(Image.open(io.BytesIO(grab_frame_png(path, t))).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            log.warning("  frame at %.2fs failed (%s) — skipping", t, exc)
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")

    batch = torch.stack([preprocess(f) for f in frames]).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)     # per-frame unit norm
        pooled = feats.mean(dim=0)
        pooled = pooled / pooled.norm()                       # re-normalise after pooling
    return pooled.detach().cpu().numpy().astype(np.float32), len(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed clips with CLIP.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="re-embed clips that already have vectors")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true",
                    help="write RANDOM vectors instead of CLIP embeddings, to exercise "
                         "the pipeline before you have GPU access. Results computed from "
                         "these are meaningless and are tagged as such in the database.")
    args = ap.parse_args()

    require_binaries("ffmpeg")
    cfg = load_config(args.config)
    e = cfg["embed"]
    device = pick_device(args.device or e.get("device", "auto"))
    n_keyframes = int(e.get("n_keyframes", 8))

    conn = connect(cfg)
    sql = """
        SELECT c.clip_id, c.path, c.duration_s
        FROM clips c
        LEFT JOIN clip_embeddings ce ON ce.clip_id = c.clip_id
        {where}
        ORDER BY c.clip_id
    """.format(where="" if args.force else "WHERE ce.clip_id IS NULL")
    with conn.cursor() as cur:
        cur.execute(sql)
        todo = cur.fetchall()
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        log.info("Nothing to embed. All clips have vectors (use --force to redo).")
        conn.close()
        return

    if args.smoke_test:
        log.warning("=" * 70)
        log.warning("SMOKE TEST: writing RANDOM vectors, not CLIP embeddings.")
        log.warning("Any retrieval metric computed from these is meaningless.")
        log.warning("Re-run without --smoke-test before evaluating anything.")
        log.warning("=" * 70)
        model = preprocess = torch = None
        load_s = 0.0
    else:
        model, preprocess, torch, load_s = load_encoder(e["model"], e["pretrained"], device)

    log.info("Embedding %d clips × %d keyframes on %s…", len(todo), n_keyframes, device)
    t0 = time.perf_counter()
    n_frames_total = 0

    rng = np.random.default_rng(0)
    for clip_id, path, duration_s in todo:
        if args.smoke_test:
            vec = rng.standard_normal(int(e["dim"])).astype(np.float32)
            vec /= np.linalg.norm(vec)
            n_used = 0
        else:
            vec, n_used = embed_clip(path, float(duration_s or 8.0), n_keyframes,
                                     model, preprocess, torch, device)
        if vec.shape[0] != int(e["dim"]):
            raise SystemExit(
                f"Embedding dim {vec.shape[0]} != config embed.dim {e['dim']}. "
                "Update both config.yaml and the VECTOR(n) column in sql/schema.sql."
            )
        n_frames_total += n_used
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO clip_embeddings (clip_id, embedding, dim, model, n_keyframes)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (clip_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding, dim = EXCLUDED.dim,
                    model = EXCLUDED.model, n_keyframes = EXCLUDED.n_keyframes,
                    created_at = now()
                """,
                (clip_id, vec, int(vec.shape[0]),
                 "SMOKE-TEST-RANDOM-VECTORS-NOT-VALID" if args.smoke_test
                 else f"{e['model']}/{e['pretrained']}", n_used),
            )
        log.info("  clip %-4s  %d frames  ok", clip_id, n_used)

    elapsed = time.perf_counter() - t0
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM clip_embeddings")
        total = cur.fetchone()[0]
    conn.close()

    stats = {
        "device": device,
        "model": ("SMOKE-TEST-RANDOM-VECTORS-NOT-VALID" if args.smoke_test
                  else f"{e['model']}/{e['pretrained']}"),
        "smoke_test": bool(args.smoke_test),
        "dim": int(e["dim"]),
        "n_keyframes_per_clip": n_keyframes,
        "clips_embedded": len(todo),
        "frames_encoded": n_frames_total,
        "model_load_s": round(load_s, 2),
        "embed_wall_clock_s": round(elapsed, 2),
        "seconds_per_clip": round(elapsed / len(todo), 3),
        "ms_per_frame": round(1000 * elapsed / max(n_frames_total, 1), 1),
        "total_embeddings_in_db": total,
    }
    log.info("Embedded %d clips in %.1fs (%.0f ms/frame).", len(todo), elapsed, stats["ms_per_frame"])
    out = write_json(cfg.get_path("paths.outputs_dir") / "embed_stats.json", stats)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
