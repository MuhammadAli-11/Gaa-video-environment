"""Shared plumbing: config loading, database connections, ffmpeg wrappers.

Kept deliberately small. Every script imports from here so that paths, database
credentials and encoder settings have exactly one definition.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    # A Windows console defaults to cp1252, which cannot encode the non-ASCII
    # characters these scripts log (≈, →, ×). That raises inside the logging
    # handler, printing a traceback in the middle of a run for a cosmetic
    # reason. Force UTF-8 on the stream where the platform allows it.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


log = logging.getLogger("common")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class Config(dict):
    """dict with dotted access: cfg.get_path('paths.clips_dir')."""

    def dot(self, key: str, default: Any = None) -> Any:
        node: Any = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get_path(self, key: str, default: Any = None) -> Path:
        raw = self.dot(key, default)
        if raw is None:
            raise KeyError(f"config key not found: {key}")
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    def opt_path(self, key: str) -> Path | None:
        """Path for an OPTIONAL input, or None when the key is absent or null.

        `tracks_parquet` and `manifest` are genuinely optional: an explicit
        `null` means "this input does not exist for this corpus", which is a
        different statement from "someone forgot to configure it". get_path()
        cannot distinguish them, so it raised on both.
        """
        raw = self.dot(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path) as fh:
        data = yaml.safe_load(fh)
    # Environment overrides for the database, so the same config works in CI.
    db = data.setdefault("db", {})
    for env_key, cfg_key in [
        ("PGHOST", "host"), ("PGPORT", "port"), ("PGDATABASE", "name"),
        ("PGUSER", "user"), ("PGPASSWORD", "password"),
    ]:
        if os.environ.get(env_key):
            db[cfg_key] = os.environ[env_key]
    return Config(data)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def dsn(cfg: Config) -> str:
    d = cfg["db"]
    return (
        f"host={d['host']} port={d['port']} dbname={d['name']} "
        f"user={d['user']} password={d['password']}"
    )


def connect(cfg: Config, autocommit: bool = True):
    """Open a connection to whichever backend config.db.backend names.

    sqlite   → stdlib, no server, schema created on first call
    postgres → psycopg3 with the pgvector adapter registered
    """
    import store

    if store.backend(cfg) == "sqlite":
        return store.init_sqlite(cfg)

    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn(cfg), autocommit=autocommit)
    register_vector(conn)
    ef = cfg.dot("index.ef_search", 40)
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {int(ef)}")
    return conn


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe
# ---------------------------------------------------------------------------
def _resolve_binary(name: str) -> str | None:
    """Find ffmpeg/ffprobe on PATH, or fall back to the pip-installed build.

    A system ffmpeg is preferred when present. `imageio-ffmpeg` ships a static
    ffmpeg wheel but NOT ffprobe, so ffprobe may legitimately resolve to None
    even when ffmpeg is available; `ffprobe()` handles that by probing with
    OpenCV instead of failing.
    """
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001 — package absent or no bundled binary
            return None
    if name == "ffprobe":
        # some distributions place ffprobe beside ffmpeg
        ff = _resolve_binary("ffmpeg")
        if ff:
            cand = Path(ff).with_name("ffprobe" + Path(ff).suffix)
            if cand.exists():
                return str(cand)
    return None


FFMPEG = _resolve_binary("ffmpeg")
FFPROBE = _resolve_binary("ffprobe")
_BINARIES = {"ffmpeg": FFMPEG, "ffprobe": FFPROBE}


def require_binaries(*names: str) -> None:
    # ffprobe is not required: probe_media() falls back to OpenCV, which reads
    # the same container metadata through the same libav under the hood.
    missing = [n for n in names if n != "ffprobe" and _BINARIES.get(n, shutil.which(n)) is None]
    if missing:
        raise RuntimeError(
            f"Required binaries not found: {', '.join(missing)}. "
            "Install ffmpeg system-wide, or `pip install imageio-ffmpeg` for a "
            "bundled build, and re-run."
        )


def run(cmd: Iterable[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    # Call sites say "ffmpeg"; swap in the resolved path so a bundled binary
    # works without anyone having to touch PATH.
    if cmd and cmd[0] in _BINARIES and _BINARIES[cmd[0]]:
        cmd[0] = _BINARIES[cmd[0]]
    log.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture)


def _fraction(text: str | None) -> float | None:
    if not text or "/" not in text:
        try:
            return float(text) if text else None
        except ValueError:
            return None
    num, den = text.split("/", 1)
    try:
        den_f = float(den)
        return float(num) / den_f if den_f else None
    except ValueError:
        return None


@dataclass
class MediaInfo:
    path: Path
    duration_s: float
    fps: float | None
    n_frames: int | None
    width: int | None
    height: int | None
    codec: str | None
    bytes: int

    def as_row(self) -> dict:
        return {
            "filename": str(self.path),
            "duration_s": self.duration_s,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "bytes": self.bytes,
        }


_OPENCV_PROBE_NOTED = False


def _probe_opencv(path: Path) -> MediaInfo:
    """Container metadata without ffprobe.

    OpenCV reads it through the same libav the ffprobe binary wraps, so the
    numbers agree. Codec name is reported via the FOURCC tag, which is coarser
    than ffprobe's `codec_name` — that is the only thing lost here.
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path} with OpenCV and ffprobe is unavailable")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip() or None
    duration = (n_frames / fps) if (n_frames and fps) else 0.0
    global _OPENCV_PROBE_NOTED
    if not _OPENCV_PROBE_NOTED:
        log.warning("ffprobe not found — probing with OpenCV instead "
                    "(same libav underneath; only codec_name is coarser). "
                    "Logged once per process.")
        _OPENCV_PROBE_NOTED = True
    return MediaInfo(path=path, duration_s=duration, fps=fps, n_frames=n_frames,
                     width=w, height=h, codec=codec, bytes=path.stat().st_size)


def ffprobe(path: str | Path) -> MediaInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if FFPROBE is None:
        return _probe_opencv(path)
    proc = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    meta = json.loads(proc.stdout)
    video = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = meta.get("format", {})

    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    n_frames = video.get("nb_frames")
    n_frames = int(n_frames) if n_frames and str(n_frames).isdigit() else (
        int(round(duration * fps)) if (duration and fps) else None
    )
    return MediaInfo(
        path=path,
        duration_s=duration,
        fps=fps,
        n_frames=n_frames,
        width=int(video["width"]) if video.get("width") else None,
        height=int(video["height"]) if video.get("height") else None,
        codec=video.get("codec_name"),
        bytes=path.stat().st_size,
    )


def grab_frame_png(path: str | Path, t_s: float) -> bytes:
    """Decode a single frame at t_s and return it as PNG bytes.

    `-ss` before `-i` uses input seeking, which is near-instant on the short
    clips this pipeline produces.
    """
    proc = run([
        "ffmpeg", "-nostdin", "-v", "error",
        "-ss", f"{max(t_s, 0):.3f}", "-i", str(path),
        "-frames:v", "1", "-f", "image2", "-vcodec", "png", "-",
    ])
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg returned no frame for {path} at t={t_s:.3f}s")
    return proc.stdout


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def dir_bytes(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human_mb(n_bytes: float) -> float:
    return round(n_bytes / (1024 ** 2), 2)


def pick_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch not installed yet
        pass
    return "cpu"
