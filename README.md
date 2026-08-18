# gaa-video-environment

An AI-first video environment for match-day analysis: ingestion, event indexing and
semantic retrieval over Gaelic football clips, **built to a latency budget rather
than an accuracy target**.

Half-time lasts fifteen minutes. Within it an analyst has perhaps ninety seconds of
real attention for any one question. That constraint drives every design decision
here:

- **All expensive work happens at ingest.** Probing, proxy generation, clip cutting
  and embedding are done as footage arrives.
- **Query time is cheap by construction.** One text encode, one cosine ANN lookup.
  Nothing on the request path decodes video or runs a detector.
- **Retrieval is hybrid, and that is the dependency worth measuring.** Structured
  metadata from the vision layer pre-filters the candidate set; vector similarity
  ranks what survives. Pure semantic search over sport video is not good enough on
  its own — but the pre-filter inherits whatever the vision layer got wrong. Measured
  here: a tracker that fragments identity produces a player count inflated **3.5×**,
  which turns `n_players >= 12` into a filter that admits **18 of 19 clips instead of
  4** and reports no error while doing it. See [The finding](#the-finding).

Companion to `gaa-kickout-vision` (Project 1). The two share **tooling and
findings, not footage** — see below, because the distinction changes what may be
concluded from either.

## Current corpus, and what it does not share with Project 1

`data_2.mp4` — an edited compilation, 12m35s, 854×478 upscaled to 1280×720,
~29.7 fps, **19 kickouts**.

**This is not the footage Project 1 evaluates on.** Project 1's measured
baseline comes from `lgf26_final_w1`: a Ladies Gaelic Football final, natively
1280×720, from a 28 fps broadcast source, annotated and evaluated as a
continuous passage of one match. `data_2` is a **men's Gaelic football
compilation** — a different code with different kick range and rules, at roughly
a third of the pixel area, cut and edited rather than continuous.

Consequences, stated because they are easy to forget once both repos are open:

| | Project 1 (`lgf26_final_w1`) | This repo (`data_2`) |
|---|---|---|
| Code | Ladies GF | Men's GF |
| Native resolution | 1280×720 | 854×478 (upscaled) |
| Continuity | continuous match passage | edited compilation, cuts and replays |
| Role | detection/tracking evaluation | retrieval corpus |

So Project 1's accuracy figures **do not transfer** to this corpus as
predictions — they are measurements on other footage, quoted here to explain
*mechanisms* rather than to assert what the detector achieves on `data_2`. What
genuinely crosses over is the **code** (`s02`/`s03` are run here directly) and
the **failure modes** those figures established.

What Project 1 does supply for this corpus:

- the **segmentation** of `data_2` and the **19 hand-marked kickout timestamps**,
  bridged into compilation time by `tools/events_from_project1.py`. These are
  deliberately *not* `s07`'s predictions: that detector measured precision@8 =
  1/8 with an inverted velocity term, so indexing it would confound retrieval
  error with detection error.
- the **detector and tracker themselves**, run on `data_2` via
  `tools/detect_players_data2.py` to populate `n_players_in_contest`.

Measured end to end on CPU: ingest **4.95× realtime** (7.1 min for a 35-min half),
warm query **p50 86 ms**, of which vector search is **0.39 ms**. Full numbers and
the argument they support: [`report/system_report.md`](report/system_report.md).

---

## What is in here

```
├── config.yaml               every tunable in one place, incl. db.backend
├── docker-compose.yml        optional: Postgres 16 + pgvector
├── sql/schema_sqlite.sql     default backend, created automatically
├── sql/schema.sql            Postgres equivalent
├── src/
│   ├── common.py             config, DB, ffmpeg helpers
│   ├── store.py              backend switch + the NumPy ranking index
│   ├── s00_ingest.py         probe → assets row, proxy, timecode alignment check
│   ├── s01_extract_clips.py  events_pred.csv → padded clips + thumbnails + DB rows
│   ├── s02_embed.py          CLIP keyframe embeddings (the only GPU step)
│   ├── s03_build_index.py    index build + the exhaustive/ANN crossover sweep
│   ├── s04_api.py            FastAPI: /search /events /clip /warmup /session
│   ├── s05_benchmark.py      latency, retrieval quality, synthetic scale test
│   ├── s06_pilot_analysis.py time-on-task, success rate, NASA-TLX, figures
│   └── s07_stats.py          paired significance tests, bootstrap CIs, effect sizes
├── ui/index.html             single file, no framework, countdown timer for pilots
├── pilot/                    protocol, tasks, TLX, consent, response templates
├── eval/                     relevance judgements for the retrieval evaluation
├── tools/
│   ├── events_from_project1.py  bridge Project 1's hand-marked kickouts into
│   │                            compilation time, with a timebase cross-check
│   ├── detect_players_data2.py  run Project 1's s02/s03 on data_2 to populate
│   │                            n_players_in_contest (de-duplicated, not track ids)
│   └── make_demo_events.py      stand-in events for building before Project 1 lands
└── report/system_report.md   the three-page deliverable
```

---

## Setup

### Prerequisites

- Python 3.10+
- **ffmpeg** — either on `PATH`, or `pip install imageio-ffmpeg` for a bundled
  static build, which `src/common.py` resolves automatically.
- **ffprobe** is optional. If it is absent, container metadata is read with
  OpenCV instead — the same libav underneath, only the codec name is coarser.

That is the whole list. **No database server, no Docker.**

```bash
git clone <your-repo-url> gaa-video-environment
cd gaa-video-environment

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The default `db.backend: sqlite` stores everything in `data/gaa.db` and ranks in
NumPy. The schema is created on first use; there is nothing to start and nothing to
apply.

### Why SQLite is not a downgrade

Ranking is an exhaustive cosine over a contiguous float32 matrix — one BLAS call.
Measured on this repo (`s03_build_index.py`, random unit vectors, the pessimal case
for any index):

| Corpus | Median search | Resident |
|---|---|---|
| 19 clips (this corpus) | **0.03 ms** | 0.04 MB |
| 1,000 | 0.13 ms | 1.9 MB |
| 10,000 (a season) | 1.25 ms | 19.5 MB |
| 100,000 | 13.3 ms | 195 MB |
| 1,000,000 | 137 ms | 1,953 MB |

Against a 90-second task budget, exhaustive search is free up to about 100,000
clips — several seasons of footage. An HNSW probe cannot beat 0.03 ms, and a
round-trip to a database socket to attempt it costs more than the arithmetic. The
approximate index is architecture for a scale this project does not reach, and
`s03` measures that rather than assuming it.

Past 100,000, **memory binds before latency does**: at 1M vectors the 137 ms search
still fits the budget but the 1.9 GB resident matrix does not fit comfortably in a
laptop process. That is where an on-disk index earns its place, and not before.

Exhaustive search is also **exact**, which removes a whole class of caveat from the
evaluation: there is no recall/latency tradeoff to tune and no approximation error
confounding the semantic-versus-hybrid comparison.

### If you do want Postgres

Everything still works. Uncomment the database lines in `requirements.txt`, set
`db.backend: postgres` in `config.yaml`, and `docker compose up -d`. The evaluation
produces the same numbers; only the ranking implementation changes. Being able to
say *"I measured both and chose the simpler one for this scale"* is a stronger
answer than having used Postgres by default.

> **GPU note.** `requirements.txt` pins the default CPU build of torch. For the A100,
> install the matching CUDA wheel *before* the requirements file:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
> Only `s02_embed.py` needs the GPU, and it needs it for about a minute.

### Point at your inputs

Edit `config.yaml`:

```yaml
paths:
  source_video: data/raw/match.mp4                              # from Project 1's s00_prepare_footage.py
  events_csv:   ../gaa-kickout-vision/outputs/events_pred.csv
  tracks_parquet: ../gaa-kickout-vision/outputs/tracks.parquet
  manifest:     ../gaa-kickout-vision/outputs/manifest.json    # optional, for the timecode check
```

`s01` accepts several column spellings for the events CSV. It needs start and end
times at minimum; `t_peak_s`, `confidence`, `n_players` and `pitch_zone` are used
when present. If `pitch_zone` is absent it is derived from `tracks.parquet` when
that file carries normalised pitch coordinates, and left `NULL` otherwise — a wrong
zone label is worse than a missing one, because it is a pre-filter and a mislabelled
clip becomes permanently unreachable.

**No Project 1 output yet?** Build the plumbing against stand-in events:

```bash
python tools/make_demo_events.py --n 15
python src/s01_extract_clips.py --events data/demo_events_pred.csv
```

Synthetic timestamps are fine for plumbing and useless for evaluation. Swap them out
before you measure anything.

---

## Run order

```bash
# 0. No setup step. The database is created on first use.

# 0. Bridge Project 1's events into this repo's timebase
python tools/events_from_project1.py

# 1. Register the asset, build the scrubbing proxy, verify timecode alignment
python src/s00_ingest.py

# 2. Cut one padded clip + thumbnail per detected event
python src/s01_extract_clips.py

# 3. Embed clips — the only GPU step, ~1 minute for 15 clips
python src/s02_embed.py

# 4. Build the ranking index and sweep the exhaustive/ANN crossover
python src/s03_build_index.py

# 5. Serve
uvicorn s04_api:app --app-dir src --port 8000
```

Then open **http://127.0.0.1:8000/**.

Or, once the database is up, run steps 1–4 in one go:

```bash
make pipeline      # or: bash run_pipeline.sh
```

### Evaluate

```bash
# with the API running in another terminal:

python src/s05_benchmark.py --init-eval     # writes eval/queries.yaml with your clip inventory
# → hand-label relevance for 15–20 queries BEFORE looking at any system output

python src/s05_benchmark.py --mode all      # latency + retrieval quality
python src/s07_stats.py                     # significance tests, CIs, effect sizes
```

`s07_stats.py` is the inferential layer and writes `outputs/statistical_report.md`
with tables you can paste straight into the write-up. What it does, and why:

| Data | n | Approach |
|---|---|---|
| Retrieval | queries (~16–20) | Paired randomisation (permutation) test on per-query differences, exact by enumeration when n ≤ 16. Percentile bootstrap CIs over queries. Cohen's d_z. Holm-Bonferroni across secondary metrics, with nDCG@10 pre-specified as primary. |
| Latency | requests (~100+) | Bootstrap CI on the p95 itself — a tail percentile from a few hundred samples is noisier than people assume. Wilson interval on the proportion under budget. |
| Pilot | participants (3–5) | **No hypothesis test.** Wilson interval computed only to show how wide it is, which argues for descriptive-only reporting better than asserting it. |

Both arms run on the same query set, so everything is paired — that is what buys
usable power at n=16. A t-test is the wrong tool here: per-query P@5 takes six
possible values and is nowhere near normal, which is why IR evaluation uses
randomisation tests.

Outputs land in `outputs/`: `benchmark.json`, `retrieval_eval.json`,
`scale_bench.json`, `index_stats.json`, and figures in `outputs/figures/`.

### Run the pilot

1. Book participants **first**. Availability is the only dependency you do not
   control and the thing most likely to slip.
2. Read `pilot/protocol.md` end to end, and `pilot/tasks.md` during the session.
3. Start the API, fire `curl -X POST localhost:8000/warmup`, and share your screen.
4. In the UI: enter a participant id → **Start session** → pick a task →
   **Start task** → **Found it** / **Give up**. Queries, clip opens and task
   boundaries are logged server-side, so time-on-task is measured rather than
   stopwatched.
5. Enter TLX responses in `pilot/tlx_responses.csv`, themes in `pilot/themes.csv`.
6. `python src/s06_pilot_analysis.py`

---

## API

| Endpoint | Behaviour |
|---|---|
| `GET /search?q=&event_type=&zone=&limit=&mode=` | text encode → cosine ANN, optional SQL pre-filter. `mode=semantic` disables filters entirely — the ablation arm |
| `GET /events?event_type=&zone=&t_from=&t_to=` | structured filtering only, no embedding — the fast path |
| `GET /clip/{id}` | streams the clip with HTTP range support (206) |
| `GET /thumb/{id}` | thumbnail |
| `GET /facets` | vocabulary the index actually knows, for the UI to show |
| `GET /health` | liveness; deliberately does **not** load the model, so cold start stays measurable |
| `POST /warmup` | loads the text tower and runs one throwaway encode |
| `POST /session/start`, `/session/event`, `/session/{id}/end` | pilot instrumentation |

Every `/search` and `/events` call is written to `query_log` with its latency
breakdown, which is what makes the numbers in the report measured rather than
asserted.

```bash
curl "localhost:8000/search?q=players+jumping+for+a+high+ball&limit=5" | jq
curl "localhost:8000/search?q=contested+kickout&mode=semantic" | jq '.timings_ms'
curl "localhost:8000/events?zone=middle&limit=20" | jq '.n_results'
```

---

## The finding

> Pure semantic retrieval over sport video underperforms because general-purpose
> vision-language models have no sport-specific grounding. Structured event metadata
> from the vision layer is what makes retrieval usable — so the CV layer and the data
> architecture are not separable concerns. They are one design problem.

On the `data_2` corpus, one field has been populated by running Project 1's own
`s02`/`s03` over all 19 clips — and it demonstrates something sharper than absence:
**fragmentation supplies corrupted metadata, not missing metadata.**

| | Naive (distinct track IDs) | De-duplicated (median simultaneous detections/frame) |
|---|---|---|
| Median | **31** | **9** |
| Range | 7 – 71 | 2 – 21 |
| Clips reading > 30 "players" | **11 / 19** | 0 / 19 |

A kickout contest involves at most ~15 outfield players near the ball, so 11 of 19
naive counts are physically impossible. Inflation: **median 3.5×, range 2.4–10.0×**.

Rescaling cannot fix it. The naive count *ranks* clips fine (Spearman +0.812) but
its **scale** is wrong, and a filter thresholds absolute values. The inflation also
varies 4-fold and anti-correlates with the quantity being measured (ρ = −0.663 against
players actually present) — worst on sparse close-ups, mildest on crowded wide shots.

The consequence is a filter that is a no-op wearing a filter's name:

| Threshold | Naive passes | De-duplicated passes |
|---|---|---|
| `n_players >= 4` | **19/19** | 16/19 |
| `n_players >= 12` | **18/19** | **4/19** |
| `n_players >= 20` | 17/19 | 3/19 |

It returns no error and no warning — just almost everything.

With the de-duplicated field live, the six-query comparison now gives **3/6 identical**
(was 6/6), mean Jaccard 0.625. One weak field was enough to separate the arms.

**But the divergence may be harmful.** The three queries that changed are the three
about sparse phases — a keeper restarting, a ball downfield, a short kick — and the
filter dropped the same three low-count clips from each. Those are plausibly the most
relevant answers for "goalkeeper restart". Without relevance judgements this repo
**does not claim the divergence is an improvement**, only that it is real.

A zone could be approximated rather than left NULL. It is refused on purpose:
`pitch_zone` is a **pre-filter**, so a wrong label makes a clip permanently
unreachable. A missing zone costs a filter; a wrong zone costs a clip, silently.

**No P@5 / MRR / nDCG figure is reported.** Relevance judgements must be made
before seeing system output or they measure the labeller's anchoring, and with one
event class "find a kickout" retrieves 19 of 19 by construction. `eval/queries.yaml`
is generated and ready for a human; `s07_stats.py` is implemented and has nothing
valid to run on yet.

### Why clips are re-encoded

A worked example from a 40-second 25 fps test clip with a 250-frame GOP, cutting an
8-second window:

| Mode | Wall clock | Clip duration produced |
|---|---|---|
| `stream_copy` | 56 ms | **10.00 s** — snapped to the nearest keyframe |
| `reencode` | 640 ms | 8.00 s — exact |

Stream copy is 11× faster and two seconds wrong, and the error scales with GOP
length. Two seconds is the difference between a clip that opens on the kick and one
that opens after it. Once an analyst catches a timestamp lying to them, they stop
trusting all of them, and a tool nobody trusts is not a faster tool. Both modes are
implemented (`clips.mode`) and `s05` reports the throughput cost of the choice.

Two further pieces of honesty the code enforces rather than hides:

- **The ANN index does not matter at single-match scale.** With tens of clips
  Postgres correctly sequentially scans, and `s03` reports that instead of pretending
  otherwise. `s05 --mode scale` finds where HNSW starts earning its keep on
  synthetic data — the index is architecture for a season, not for a match.
- **Cold start is reported, not averaged away.** The first query of every half-time
  is cold. `/health` deliberately does not warm the model so that the cost stays
  measurable, and `/warmup` exists to pay it while the teams are still walking off.

---

## Sanity ranges

If your numbers land far outside these, something is wrong — check before you write
them up.

| Metric | Plausible | If it is off |
|---|---|---|
| Query latency p95, warm | 80–400 ms | over 1 s means you are embedding at query time wrongly |
| Cold start | 2–8 s on GPU; **30.8 s measured on CPU** | model load dominates (30.7 s of it). Always `/warmup` before half-time — an unwarmed first query eats a third of the 90 s budget |
| Ingest throughput | 2–6× realtime on CPU | measured 4.95× → a 35-min half in 7.1 min |
| Storage per match-minute | 15–60 MB | measured 7.34 MB on 854×478 source; scales with resolution |
| P@5, pure semantic | 0.30–0.55 | not measured on this corpus — see [The finding](#the-finding) |
| P@5, hybrid | 0.55–0.80 | not measured. The filter now exists and *changes* results (3/6 queries), but whether it improves them needs relevance judgements |
| MRR | 0.4–0.7 | not measured |
| Task success rate | 60–85% | |
| Median time on task | 40–120 s | some tasks will blow the 90 s budget |

---

## Troubleshooting

**`database is locked`** — another process holds the SQLite write lock. Stop the
API before running `s01`/`s02`, which write; read-only scripts are fine alongside it.

**Want to start over** — `rm data/gaa.db*`. The schema is recreated on the next run.

**Postgres backend: `connection refused` on 5432** — `docker compose ps`; wait for
`healthy`, not `running`. `extension "vector" does not exist` means you are on a
plain `postgres` image rather than `pgvector/pgvector:pg16`.

**`Embedding dim 768 != config embed.dim 512`** — you changed the encoder. Update `embed.dim` in
`config.yaml`. On sqlite the dimension is stored per row and nothing else needs
changing; on Postgres also update the `VECTOR(n)` column in `sql/schema.sql`.

**`Required binaries not on PATH: ffmpeg`** — install ffmpeg; ffprobe ships with it.

**Clips start in the wrong place** — you are in `stream_copy` mode, which snaps to
keyframes. Use `clips.mode: reencode` (the default).

**API returns zero results** — `curl localhost:8000/health` and check
`indexed_clips`. If it is 0, `s02_embed.py` has not run. The index is loaded at
startup, so restart the API after embedding.

**`health` shows `"smoke_test": true`** — `GAA_SMOKE_TEST` is set in your
environment. Unset it, or every number you produce is noise.

**Cold-start benchmark hangs** — it launches a second uvicorn on port 8099. If that
port is busy, pass `--cold-port`, or skip with `--skip-cold`.

---

## Notes on scope

Single match, single event class, one camera angle. No ball tracking, no score
state, no game clock. Pilot participants were football-literate testers rather than
practising performance analysts, so their mental model of half-time is inferred
rather than lived — validating with elite analysts is the obvious next step and the
one that needs access to applied partners.

Raw participant data lives in `pilot/sessions/`, which is gitignored and never
committed.
