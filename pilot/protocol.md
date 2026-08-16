# Pilot usability protocol

**Study question.** Can a football-literate person find a specific piece of footage
inside the ninety seconds an analyst realistically has during half-time, and what
does the attempt cost them in workload?

**Design.** Within-subjects, single condition, think-aloud. 3–5 participants,
15 minutes each, screen-shared and recorded with consent.

**This is a pilot, not a study.** It is here to surface interaction problems and to
check that the instrumentation works, not to produce a statistically defensible
claim. Everything is reported descriptively.

---

## Before the session

- [ ] Participants booked. This is the only dependency you do not control.
- [ ] `docker compose up -d` and the API running.
- [ ] `POST /warmup` fired, or the first participant pays the model-load cost and
      you measure your own cold start instead of their task.
- [ ] Consent form sent and returned (`consent_form.md`).
- [ ] Recording tool tested. Test the audio, not just the video.
- [ ] `pilot/tasks.md` open in a separate window so you can read tasks verbatim.

## Session structure (15 minutes)

| Minutes | Segment |
|---|---|
| 0–2 | Consent check, recording starts, what the system is and is not |
| 2–4 | Orientation: one worked example task, unrecorded, questions allowed |
| 4–12 | Four timed tasks, think-aloud, no help from the facilitator |
| 12–14 | NASA-TLX |
| 14–15 | Open debrief: what was confusing, what was missing |

## Framing script (read this, do not improvise)

> This is a tool for finding moments in match footage. Imagine you are an analyst
> at half-time: you have about ninety seconds to find a clip before the coach needs
> it. I will read you a task, you start the timer, and you talk through what you are
> doing and thinking as you go. There are no wrong moves — if something is confusing,
> that is information about the tool, not about you. I will not answer questions
> during a task, but ask them freely afterwards.

## Facilitator rules

- Do not rescue. When a participant stalls, that stall is the result.
- If asked a direct question during a task: *"What would you do if I were not here?"*
- Prompt for think-aloud only when someone goes quiet for ~10 seconds:
  *"What are you thinking now?"* Never *"Why did you do that?"* mid-task — it turns
  the participant into an analyst of their own behaviour.
- Note the exact words participants use for what they are looking for. Vocabulary
  mismatch is the failure mode you most expect to see, and their phrasing is the
  data.
- Stop a task at 180 seconds (twice the budget). Record it as abandoned. Let them
  finish the thought, then move on.

## Instrumentation

The UI logs to the database. Enter the participant id, press **Start session**,
select the task, press **Start task**, and press **Found it** or **Give up** at the
end. Every query, clip open and task boundary is timestamped server-side, so
time-on-task is measured rather than stopwatched.

Verify after each session:

```bash
psql -h localhost -U postgres -d gaa -c \
  "SELECT participant, task_id, kind, round(t_ms::numeric/1000,1) AS s
   FROM pilot_events pe JOIN pilot_sessions s USING (session_id)
   ORDER BY pe_id DESC LIMIT 20;"
```

If the rows are not there, the session data is gone and no amount of careful
analysis afterwards will bring it back. Check every time.

## After all sessions

1. Enter TLX responses into `pilot/tlx_responses.csv`.
2. Code the think-aloud transcripts into `pilot/themes.csv`
   (`theme,participant,note`). Aim for 3–5 themes; do not invent a sixth to look
   thorough.
3. `python src/s06_pilot_analysis.py`

## Ethics and data handling

- Participation is voluntary, unpaid and can be stopped at any point without
  giving a reason.
- Recordings and raw notes live in `pilot/sessions/`, which is gitignored, and are
  deleted once the themes are coded.
- Participants are identified by a code (P1, P2…) everywhere except the consent
  forms.
- Only anonymised, aggregate results appear in the write-up.
- This pilot involves human participants. Before running anything comparable
  inside a university programme, check with the institutional research ethics
  committee — a pilot done for a portfolio does not need approval in the same way
  a study does, but knowing where that line sits is itself part of the job.
