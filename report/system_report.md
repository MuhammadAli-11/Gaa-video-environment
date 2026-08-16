# gaa-video-environment — system report

**Corpus:** `data_2.mp4`, an edited Gaelic football compilation — 12m35s,
854×478, ~29.7 fps, 91.2 MB, 19 kickouts.
**Companion:** `gaa-kickout-vision` (Project 1), which supplied the
segmentation and the event timestamps.
**Generated:** 2026-08-16. All figures below are measured on this machine
(CPU-only, no GPU) unless stated.

---

## 1. What was built

An ingest-heavy, query-light retrieval environment over match video. Every
expensive operation — probing, proxy generation, clip cutting, embedding —
happens once at ingest. A query costs one CLIP text encode plus one cosine
ranking pass. Nothing on the request path decodes video or runs a detector.

| Stage | Script | Result on this corpus |
|---|---|---|
| Ingest | `s00_ingest.py` | 1 asset, 360p proxy in 70.2 s (**10.75× realtime**), 31.9 MB |
| Clip cutting | `s01_extract_clips.py` | 19 clips, 8 s each, 38.5 s (**19.6× realtime**), 15.5 MB |
| Embedding | `s02_embed.py` | 19 clips × 8 keyframes = 152 frames, 43.8 s (288 ms/frame, CPU) |
| Index | `s03_build_index.py` | 19 × 512 float32 = 0.04 MB, load 0.29 ms |
| API | `s04_api.py` | FastAPI, 19 clips indexed |

Total ingest **152.5 s for 12m35s of footage — 4.95× realtime**, which
extrapolates to **7.1 minutes to ingest a 35-minute half**. Storage is
7.34 MB per match-minute, or **0.251 GB per half** including proxy, clips
and thumbnails.

---

## 2. Where the events came from, and why it matters

Project 1 can emit predicted events, and this project originally indexed
them. That was changed deliberately.

Project 1's own evaluation puts its rule detector at **precision@8 = 1/8**,
and found its velocity term was scoring contests *backwards* — the
broadcast frames a kickout wide, so pixel velocity is lower at a contest
than in open play (`docs/10 framing_scale`). Indexing those predictions
would fold detector error into every retrieval number here, and a retrieval
metric that cannot separate *"the ranker failed"* from *"that clip was not
a kickout"* measures nothing.

Project 1 also produced **19 hand-marked kickout timestamps** for `data_2`,
one skeleton per extracted segment. `tools/events_from_project1.py` bridges
those into compilation time. They are still a Project 1 artefact — the
segmentation that produced them is Project 1 work — but they carry no
detector error, so retrieval metrics computed over them measure retrieval.

Two clocks exist and confusing them silently shifts every clip:
segment-local time (0 at each segment start) and compilation time (0 at the
start of `data_2.mp4`). The bridge re-derives one from the other for every
event and refuses to write if any disagrees by more than 0.5 s. All 19
passed.

---

## 3. Latency — the design target

The budget is 90 seconds of analyst attention at half-time. Measured over
100 warm requests:

| | p50 | p95 | p99 |
|---|---|---|---|
| Client end-to-end | 86.5 ms | 101.1 ms | 124.4 ms |
| Server total | 82.2 ms | 97.1 ms | 118.7 ms |
| **Text encode** | **81.6 ms** | 96.5 ms | 118.2 ms |
| **Vector search** | **0.39 ms** | 0.78 ms | 1.48 ms |
| Structured-only (`/events`) | 4.24 ms | 16.6 ms | 21.9 ms |

**Text encoding is 99% of server time.** Vector search is 0.39 ms — two
orders of magnitude below the encode it waits on. At three queries per
task the system consumes **0.38% of the 90-second budget**, leaving 89.7 s
for the analyst to think.

This is the number that decides the architecture. Replacing exhaustive
NumPy ranking with Postgres + pgvector HNSW would optimise **0.5% of query
time** while adding a database server, a network round-trip and
approximation error. Measured, then declined.

### Scale

Exhaustive cosine over a contiguous float32 matrix, random unit vectors
(the pessimal case for any ANN index — no cluster structure to exploit):

| Corpus | p50 | p95 | Resident |
|---|---|---|---|
| 19 (this corpus) | 0.028 ms | 0.054 ms | 0.04 MB |
| 1,000 | 0.13 ms | 0.31 ms | 1.9 MB |
| 10,000 (a season) | 1.25 ms | 2.54 ms | 19.5 MB |
| 100,000 | 13.3 ms | 16.0 ms | 195 MB |
| 1,000,000 | 137 ms | 163 ms | 1,953 MB |

Exhaustive search stays under 10 ms to roughly 100,000 clips — several
seasons. Past that, **memory becomes the binding constraint before latency
does**: at 1M vectors the 137 ms search still fits the budget, but the
1.9 GB resident matrix does not fit comfortably in a laptop process. That
is where an on-disk ANN index earns its place, and not before.

### Cold start — the one number outside its expected range

| | Measured |
|---|---|
| Process boot to healthy | 5.04 s |
| **First query** | **30.8 s** |
| Model load, of that | 30.7 s |
| Second query | 111 ms |

The project's own sanity range for cold start is 2–8 s. **This is 30.8 s,
roughly 4× outside it**, because the model loads on CPU on this machine.
The mitigation is already built: `/health` deliberately does *not* load the
model, so this cost stays visible rather than being averaged away, and
`POST /warmup` exists to pay it while the teams are still walking off. Any
deployment must warm up before half-time begins — an unwarmed first query
would consume a third of the analyst's budget on its own.

---

## 4. The finding

> **Pure semantic retrieval over sport video underperforms because
> general-purpose vision-language models have no sport-specific grounding.
> Structured event metadata from the vision layer is what makes retrieval
> usable — so the CV layer and the data architecture are not separable
> concerns. They are one design problem.**

On this corpus that claim shows up in its sharpest possible form: **the
hybrid arm has nothing to be hybrid with.**

The API pre-filters on `event_type`, `pitch_zone`, `confidence` and time.
For `data_2`:

| Field | State | Why |
|---|---|---|
| `event_type` | `kickout` for all 19 | Single event class; filtering on it is a no-op |
| `pitch_zone` | **NULL for all 19** | Needs `s04` homography, which needs hand-clicked landmarks. Never run |
| `n_players_in_contest` | **NULL for all 19** | Needs `tracks.parquet` for `data_2`; Project 1 has only run `s02`/`s03` on `lgf26_final_w1` |
| `confidence` | 1.0 for all 19 | Hand-marked, so uniform by construction |

Measured consequence — six queries, hybrid mode against `mode=semantic`:

| Query | Rankings identical? | Filters applied |
|---|---|---|
| contested kickout | yes | `{}` |
| players jumping for a high ball | yes | `{}` |
| goalkeeper restart | yes | `{}` |
| long kick downfield | yes | `{}` |
| players competing in the air | yes | `{}` |
| short kickout to the side | yes | `{}` |

**6/6 identical.** The hybrid arm collapses exactly onto the semantic arm,
because the vision layer could not supply a single usable structured field.

This is not a null result, it is the thesis stated precisely. The retrieval
architecture's advantage is *contingent on the vision layer delivering
structured metadata*, and Project 1's inability to deliver any — for
reasons that are themselves measured and documented there — removes the
advantage entirely. The two projects are one design problem, and this table
is the evidence.

The obvious objection is that a zone could be derived approximately rather
than left NULL. It is refused on purpose: `pitch_zone` is a **pre-filter**,
so a mislabelled clip becomes permanently unreachable. A missing zone
costs a filter; a wrong zone costs a clip, silently.

---

## 5. Retrieval quality — deliberately not reported

`eval/queries.yaml` is generated with the 19-clip inventory and is ready
for judgement. **No P@5, MRR or nDCG figure appears in this report**, for
two reasons:

1. **Relevance judgements must be made before seeing system output**, or
   the measurement is of the labeller's anchoring rather than the
   retrieval. The template says so; producing labels after having run the
   queries above would violate it.
2. **The corpus cannot support the comparison the evaluation is designed
   around.** Every clip is the same event class, so "find a kickout"
   retrieves 19 relevant clips out of 19 and P@5 is 1.0 by construction,
   measuring nothing. The semantic-vs-hybrid contrast is unavailable for
   the reason in §4.

`s07_stats.py` — paired randomisation tests, bootstrap CIs, Holm-Bonferroni
across secondary metrics — is implemented and ready, and is the correct
inferential layer for n≈16 queries. It has nothing valid to run on yet.

What would fix this: a corpus with **multiple event classes** and at least
one populated structured field. Both are Project 1 deliverables — the first
needs an event taxonomy beyond kickouts, the second needs `s04`.

---

## 6. Honest limitations

- **Single event class, single match, single camera.** Nineteen clips is a
  plumbing corpus, not an evaluation corpus.
- **`data_2` is an edited compilation.** Project 1 dropped it as a *tuning*
  source because cuts, replays and slow motion confound threshold fitting.
  That objection does not apply to retrieval, which reads content rather
  than fitting to it — but the clips do contain replays and camera cuts,
  and a clip that opens on a replay of the previous passage is a retrieval
  failure this corpus can produce and a live match feed could not.
- **No relevance judgements**, so no retrieval-quality claim. See §5.
- **CPU only.** Text encode at 81.6 ms p50 and model load at 30.7 s both
  improve substantially on a GPU; neither changes the architectural
  conclusion, since vector search would still be ~0.4 ms.
- **Pilot not run.** `pilot/` holds the protocol, tasks, TLX and consent
  forms; `pilot_sessions` and `pilot_events` are empty. With 3–5
  participants the correct analysis is descriptive only, and `s06` is
  written that way.
- **ffprobe is unavailable on this machine**, so container metadata is read
  via OpenCV. It goes through the same libav underneath; only the codec
  name is coarser.

---

## 7. What the two projects say together

Project 1 measured how much of a published coding scheme survives contact
with broadcast video, and found the honest answer is *less than hoped, for
reasons that are properties of the footage rather than of the method*:
frame-rate conversion upstream, framing that changes scale between contest
and open play, and cuts that are not separable from pans.

Project 2 measured what a retrieval layer built on top of that costs and
delivers, and found the query path is essentially free — 0.39 ms of
ranking behind an 81.6 ms encode — while the thing that actually determines
whether retrieval is *useful* is the structured metadata the vision layer
can supply. On this corpus it could supply none, and the hybrid arm
collapsed onto the semantic one in 6 out of 6 queries.

Neither result is the one that was hoped for. Both are measured, both are
reproducible, and both point at the same conclusion the source paper
reached from the other direction: **the bottleneck is the video, and fixing
it needs a platform rather than a better model.**
