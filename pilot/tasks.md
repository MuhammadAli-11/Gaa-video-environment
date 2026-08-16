# Pilot tasks

Read each task aloud, verbatim. Do not paraphrase, and do not use the system's own
vocabulary in the wording — if you say "kickout contest" and the participant types
"kickout contest", you have tested your prompt rather than their search.

Each task has a **defined success condition** so that "success" is not the
facilitator's impression after the fact. Decide before the session what counts.

Budget: **90 seconds**. Hard stop at 180 seconds, recorded as abandoned.

---

## T0 — Orientation (not timed, not recorded as data)

> Find any moment where players are competing for the ball in the air, and play it.

Purpose: confirm the participant can search, read a result and start playback.
Answer questions freely here and nowhere else.

---

## T1 — Single specific event

> The coach wants the kickout where our midfielder wins it cleanly, without a
> contest. Find it and be ready to play it.

**Success:** the participant opens a clip that the pre-agreed key marks as an
uncontested win, and says so.
**Tests:** whether descriptive language maps onto anything the embedding understands.

---

## T2 — Filtered set

> Show me every kickout that happened in the first ten minutes.

**Success:** the participant produces the complete set (not a semantic search that
happens to return some of them) within the budget.
**Tests:** whether they discover the structured path, or force a time-bounded
question through the text box.

---

## T3 — Comparison

> The coach thinks we lost more kickouts on the left. Find two clips that let him
> compare a kickout we won with one we lost, and have both ready.

**Success:** two clips opened, both correctly categorised by the participant.
**Tests:** multi-step retrieval under time pressure, and whether the interface
supports holding one result while looking for another. It probably does not, and
that is the finding.

---

## T4 — Deliberately unanswerable

> Find the kickout that came just after we scored a point.

**Success:** *recognising within the budget that the system cannot answer this*
counts as success. Score and game clock are not in the index.
**Tests:** whether the system communicates its own limits, or lets someone burn
ninety seconds on a question it was never going to answer. This is the most
informative task in the set, and it is the one that most needs a debrief afterwards
so nobody leaves thinking they failed.

---

## T5 — Optional, if time allows

> You have twenty seconds. Get me anything from around the thirty-minute mark.

**Success:** any clip near that timestamp, opened inside 20 seconds.
**Tests:** the floor of the latency budget, and whether the interface degrades
gracefully when there is no time to think.

---

## Facilitator record sheet

| Task | Started | Ended | Outcome | Queries | Notes |
|---|---|---|---|---|---|
| T1 | | | success / abandon | | |
| T2 | | | | | |
| T3 | | | | | |
| T4 | | | | | |

The UI logs all of this automatically. Keep the paper sheet anyway — instrumentation
fails silently and notes do not.
