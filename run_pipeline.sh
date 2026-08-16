#!/usr/bin/env bash
# Cold machine to a queryable index. No database server, no Docker.
# Needs: ffmpeg on PATH, venv active, config.yaml pointing at real inputs.
set -euo pipefail

SMOKE=""
[ "${1:-}" = "--smoke-test" ] && SMOKE="--smoke-test" && \
  echo "!! SMOKE TEST: random vectors. Any metric from this run is meaningless."

echo "==> s00 ingest";        python src/s00_ingest.py
echo "==> s01 extract clips"; python src/s01_extract_clips.py
echo "==> s02 embed";         python src/s02_embed.py $SMOKE
echo "==> s03 build index";   python src/s03_build_index.py

cat <<'MSG'

Pipeline complete. Next:

    uvicorn s04_api:app --app-dir src --port 8000

then open http://127.0.0.1:8000/ and, in another terminal:

    python src/s05_benchmark.py --init-eval    # then hand-label eval/queries.yaml
    python src/s05_benchmark.py --mode all
    python src/s07_stats.py
MSG
