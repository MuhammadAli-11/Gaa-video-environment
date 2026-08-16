.PHONY: help pipeline ingest clips embed index api bench stats pilot smoke clean reset

help:
	@grep -E '^[a-z]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

ingest:    ## s00 — probe, proxy, timecode check
	python src/s00_ingest.py

clips:     ## s01 — cut clips and thumbnails
	python src/s01_extract_clips.py

embed:     ## s02 — CLIP embeddings (needs torch; GPU optional)
	python src/s02_embed.py

index:     ## s03 — build index and sweep the exhaustive/ANN crossover
	python src/s03_build_index.py

pipeline: ingest clips embed index   ## everything up to a queryable index

smoke:     ## build the pipeline with RANDOM vectors, before you have torch/GPU
	python src/s00_ingest.py
	python src/s01_extract_clips.py
	python src/s02_embed.py --smoke-test
	python src/s03_build_index.py

api:       ## serve the API and UI on :8000
	uvicorn s04_api:app --app-dir src --port 8000

bench:     ## s05 — latency + retrieval quality (API must be running)
	python src/s05_benchmark.py --mode all

stats:     ## s07 — significance tests, bootstrap CIs, effect sizes
	python src/s07_stats.py

pilot:     ## s06 — analyse pilot sessions
	python src/s06_pilot_analysis.py

clean:     ## remove derived media, keep the database
	rm -rf data/clips/* data/thumbs/* data/proxy/*

reset:     ## delete the database as well; schema is recreated on next run
	rm -f data/gaa.db data/gaa.db-wal data/gaa.db-shm
