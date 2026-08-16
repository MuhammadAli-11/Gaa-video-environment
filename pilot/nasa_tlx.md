# NASA-TLX

Administered once, immediately after the last timed task, before the debrief.
Ask about **the set of tasks as a whole**, not each task individually.

This uses **Raw TLX (RTLX)**: the six subscales, unweighted. The original weighting
procedure adds fifteen pairwise comparisons and roughly five minutes per
participant, and the literature broadly finds the raw version performs comparably.
In a fifteen-minute session that trade is worth making, but say which version you
used — "NASA-TLX" without qualification is ambiguous.

## Administration

Read the description, then ask for a number from 0 to 100 in steps of 5.

> **Mental demand** — how much mental and perceptual activity was required?
> Was the task easy or demanding, simple or complex?
> `0 = very low ————— 100 = very high`

> **Physical demand** — how much physical activity was required?
> `0 = very low ————— 100 = very high`

> **Temporal demand** — how much time pressure did you feel because of the pace at
> which the task happened?
> `0 = very low ————— 100 = very high`

> **Performance** — how successful were you in doing what you were asked to do?
> **Note the direction: 0 is perfect, 100 is failure.** Say this aloud, because it
> is the one people reverse.
> `0 = perfect ————— 100 = failure`

> **Effort** — how hard did you have to work, mentally and physically, to reach
> your level of performance?
> `0 = very low ————— 100 = very high`

> **Frustration** — how insecure, discouraged, irritated, stressed or annoyed did
> you feel?
> `0 = very low ————— 100 = very high`

## Recording

Enter responses into `pilot/tlx_responses.csv`:

```csv
participant,mental_demand,physical_demand,temporal_demand,performance,effort,frustration,notes
P1,65,5,80,45,60,55,"struggled with wording on T1"
P2,50,0,70,30,45,35,
```

`s06_pilot_analysis.py` computes RTLX as the unweighted mean of the six subscales
and plots each participant as an individual point rather than an error bar — with
four people the spread is the finding, and a standard error implies a sampling
distribution that does not exist here.

## Interpretation caution

RTLX has no universal threshold for "too high". A score is only meaningful against
a comparison: the same task on the existing workflow, the same participants on a
different interface, or published values for a comparable task. This pilot has no
comparison condition, so the honest reading is *relative across subscales* — which
dimension carries the load — rather than *absolute*. Temporal demand running high
while physical demand is near zero says something useful. A mean RTLX of 52 on its
own says almost nothing, and claiming otherwise is the fastest way to lose an
examiner's confidence.
