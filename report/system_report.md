# gaa-video-environment — system report

**Corpus:** `data_2.mp4`, an edited **men's** Gaelic football compilation —
12m35s, 854×478 upscaled to 1280×720, ~29.7 fps, 91.2 MB, 19 kickouts.
**Companion:** `gaa-kickout-vision` (Project 1), which supplied the
segmentation, the event timestamps, and the detector/tracker run here.
**Not the same footage.** Project 1 evaluates on `lgf26_final_w1` — a
**Ladies** GF final, natively 1280×720, a continuous match passage. Different
code, different resolution, edited vs continuous. Project 1's accuracy figures
are quoted below to explain *mechanisms*, never as predictions for this corpus.
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

## 4. The finding — measured, not asserted

> **Pure semantic retrieval over sport video underperforms because
> general-purpose vision-language models have no sport-specific grounding.
> Structured event metadata from the vision layer is what makes retrieval
> usable — so the CV layer and the data architecture are not separable
> concerns. They are one design problem.**

The first version of this report could only show the *absence* of that
metadata. One field has since been populated by running Project 1's own
`s02_detect.py` and `s03_track.py` over all 19 clips
(`tools/detect_players_data2.py`, 159 min on CPU), and the result is
stronger than the absence was: **fragmentation does not merely fail to
supply the field, it supplies a corrupted one.**

### 4.1 Two ways to count players, n = 19 clips

Project 1 measured ByteTrack issuing 14–20 track identities per real player
on `lgf26_final_w1`. A distinct-track-ID count is therefore a fragmentation
count, not a player count. Both are computed here; the gap is the result.

| | Naive (distinct track IDs) | De-duplicated (median simultaneous detections/frame) |
|---|---|---|
| Median | **31** | **9** |
| Range | 7 – 71 | 2 – 21 |
| Clips reading > 30 "players" | **11 / 19** | 0 / 19 |

A kickout contest involves at most ~15 outfield players near the ball, so
**11 of 19 clips carry a naive count that is physically impossible.**

**Inflation factor: median 3.5×, IQR 3.3–4.1×, range 2.4–10.0× (n = 19).**

### 4.2 Why a rescale cannot rescue it

The naive count is not noise — it ranks clips reasonably well
(Spearman +0.812, Pearson +0.898 against the de-duplicated count). Ordering
largely survives. What does not survive is **scale**, and a filter is a
threshold on absolute value, not on rank.

Nor is the inflation a constant that could simply be divided out. It varies
4-fold across clips and it varies *systematically with scene content*:

| Correlate of the inflation factor | Spearman ρ (n=19) |
|---|---|
| Players actually present | **−0.663** |
| Total detections in window | −0.606 |
| Median player size (px) | +0.318 |

Inflation is **worst on sparse, tight-framed clips** — a two-player close-up
still accumulates 16–20 track IDs (10.0×), while a crowded wide shot with 21
players accumulates 50 (2.4×). So one fixed threshold means a different
thing on every clip, in a way that is anti-correlated with the very quantity
being measured.

### 4.3 What that does to a filter

| Threshold | Naive count passes | De-duplicated passes |
|---|---|---|
| `n_players >= 4` | **19 / 19 (100%)** | 16 / 19 (84%) |
| `n_players >= 8` | 18 / 19 | 14 / 19 |
| `n_players >= 12` | **18 / 19** | **4 / 19** |
| `n_players >= 20` | 17 / 19 | 3 / 19 |

**A filter built on the naive count is a no-op that looks like a filter.**
At `>= 12` it admits 18 of 19 clips while the true count admits 4. It would
not throw an error, log a warning, or return zero results — it would
silently return almost everything, and an analyst would reasonably conclude
the filter was working and the corpus simply contained many crowded
kickouts.

This is the contingency thesis with numbers behind it: vision-layer identity
fragmentation propagates into the retrieval layer as **metadata that is
wrong in a direction the retrieval layer cannot detect.**

### 4.4 The arms now diverge — 3 of 6 queries

With `n_players` populated from the de-duplicated estimate, the six-query
comparison was re-run (`mode=semantic` vs `mode=hybrid&min_players=4`):

| Query | Identical | Jaccard | Positional overlap |
|---|---|---|---|
| contested kickout | yes | 1.00 | 1.00 |
| players jumping for a high ball | yes | 1.00 | 1.00 |
| players competing in the air | yes | 1.00 | 1.00 |
| goalkeeper restart | **no** | 0.25 | 0.00 |
| long kick downfield | **no** | 0.25 | 0.00 |
| short kickout to the side | **no** | 0.25 | 0.00 |

**3 / 6 identical** (was 6 / 6), mean Jaccard 0.625, mean positional overlap
0.500. One weak field was enough to separate the arms.

### 4.5 The divergence may well be harmful, and this corpus cannot say

The three queries that changed are the three about phases with **few players
in frame** — a keeper restarting, a ball travelling downfield, a short kick
to the side. The filter dropped the same three sparse clips from each
(`seg01_ko_001`, `seg08_ko_002`, `seg09_ko_001`, all de-duplicated count 2).

Those are plausibly the *most* relevant clips for "goalkeeper restart". A
`min_players >= 4` filter may therefore be removing correct answers from
exactly the queries it affects. **Without relevance judgements this report
cannot say whether the divergence is an improvement or a regression, and it
does not claim either.** Divergence is a measurement; quality is not.

That is itself the honest form of the finding: a structured filter derived
from a fragmenting tracker changes retrieval results, and whether it changes
them for the better is a separate question this corpus is not equipped to
answer (§5).

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
delivers. The query path is essentially free — 0.39 ms of ranking behind an
81.6 ms encode — so latency was never the risk. What actually determines
whether retrieval is *useful* is the structured metadata the vision layer
supplies, and that is where the two projects meet.

The first version of this report could only observe that the metadata was
missing. With one field now populated, the finding is sharper and worse:
**the vision layer does not fail by supplying nothing. It fails by supplying
something wrong, in a form the retrieval layer cannot detect.** A tracker
issuing 14–20 identities per player yields a player count inflated 3.5× on
median, with the inflation varying 2.4–10.0× and anti-correlated with the
quantity being measured. The rank order survives (Spearman +0.812), so any
sanity check based on correlation passes. Only the scale is destroyed — and
a filter thresholds scale, not rank. The result is `n_players >= 12`
admitting 18 of 19 clips where the truth is 4, silently.

That is the strongest form of the contingency claim these two projects can
make together. It is not that better CV would make retrieval better. It is
that **CV error does not stay inside the CV layer**: it propagates into the
data architecture as plausible-looking metadata, and a retrieval system has
no way to tell a fragmented count from a crowded one. Detection and
retrieval cannot be validated separately, because the failure crosses the
boundary between them wearing the right shape.

Neither result is the one that was hoped for. Both are measured, both are
reproducible, and both point at the same conclusion the source paper
reached from the other direction: **the bottleneck is the video, and fixing
it needs a platform rather than a better model.**
