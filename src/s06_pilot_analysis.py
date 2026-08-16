"""s06_pilot_analysis.py — descriptive analysis of the pilot sessions.

Reads instrumented interaction data from `pilot_events` (posted by the UI during
sessions) and NASA-TLX responses from pilot/tlx_responses.csv, then produces
time-on-task, success rate, queries-per-task, TLX subscale scores and figures.

Everything here is descriptive. With n=4 there are no inferential statistics worth
running, and reporting a p-value from four participants would be worse than
reporting nothing. Say that in the write-up before anyone says it to you.

NASA-TLX note: all six subscales are scored 0–100 in steps of 5, and on all six a
higher score means a heavier load — including Performance, which is anchored
"Perfect" to "Failure". The Raw TLX (RTLX) is the unweighted mean of the six,
which is standard practice and avoids the pairwise weighting procedure.

Usage:
    python src/s06_pilot_analysis.py
    python src/s06_pilot_analysis.py --tlx pilot/tlx_responses.csv
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from common import REPO_ROOT, load_config, connect, setup_logging, write_json

log = setup_logging("s06_pilot_analysis")

TLX_SUBSCALES = ["mental_demand", "physical_demand", "temporal_demand",
                 "performance", "effort", "frustration"]
SUCCESS_KINDS = {"task_success"}
FAIL_KINDS = {"task_fail", "task_abandon"}


def load_pilot_events(cfg) -> pd.DataFrame:
    conn = connect(cfg)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.session_id, s.participant, pe.task_id, pe.kind, pe.payload,
                   pe.t_ms, pe.created_at
            FROM pilot_events pe
            JOIN pilot_sessions s ON s.session_id = pe.session_id
            ORDER BY s.session_id, pe.created_at
        """)
        rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=[
        "session_id", "participant", "task_id", "kind", "payload", "t_ms", "created_at"
    ])


def summarise_tasks(events: pd.DataFrame, budget_s: float) -> pd.DataFrame:
    records = []
    for (session_id, participant, task_id), grp in events.groupby(
        ["session_id", "participant", "task_id"], dropna=True
    ):
        grp = grp.sort_values("created_at")
        starts = grp[grp.kind == "task_start"]
        ends = grp[grp.kind.isin(SUCCESS_KINDS | FAIL_KINDS)]
        if starts.empty or ends.empty:
            log.warning("Session %s task %s has no start/end pair — skipped.", session_id, task_id)
            continue
        t_start = starts.iloc[0].created_at
        end_row = ends.iloc[-1]
        duration_s = float(end_row.t_ms) / 1000 if end_row.t_ms else (
            end_row.created_at - t_start).total_seconds()

        queries = grp[grp.kind == "query"]
        clip_opens = grp[grp.kind == "clip_open"]
        errors = grp[grp.kind == "error"]

        records.append({
            "session_id": session_id,
            "participant": participant,
            "task_id": task_id,
            "success": end_row.kind in SUCCESS_KINDS,
            "time_on_task_s": round(duration_s, 1),
            "within_budget": duration_s <= budget_s,
            "n_queries": len(queries),
            "n_clip_opens": len(clip_opens),
            "n_errors": len(errors),
            "first_query": (queries.iloc[0].payload or {}).get("q") if len(queries) else None,
            "last_query": (queries.iloc[-1].payload or {}).get("q") if len(queries) else None,
        })
    return pd.DataFrame(records)


def load_tlx(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("No TLX responses at %s — skipping workload analysis.", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    missing = [c for c in TLX_SUBSCALES if c not in df.columns]
    if missing:
        raise SystemExit(f"TLX file is missing columns: {missing}. Expected: participant, "
                         + ", ".join(TLX_SUBSCALES))
    for c in TLX_SUBSCALES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].max() is not np.nan and df[c].max() > 100:
            raise SystemExit(f"TLX column {c} has values above 100 — check the scale.")
    df["rtlx"] = df[TLX_SUBSCALES].mean(axis=1).round(1)
    return df


def plot_tlx(tlx: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = [tlx[c].mean() for c in TLX_SUBSCALES]
    labels = [c.replace("_", "\n") for c in TLX_SUBSCALES]
    x = np.arange(len(TLX_SUBSCALES))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, means, color="0.75", edgecolor="black", width=0.6, label="mean")
    # Every participant plotted individually: with n=4 the spread is the finding,
    # and an error bar would imply a sampling distribution that does not exist.
    for _, row in tlx.iterrows():
        ax.scatter(x, [row[c] for c in TLX_SUBSCALES], zorder=3, s=28,
                   label=str(row.get("participant", "")))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("rating (0–100, higher = heavier load)")
    ax.set_title(f"NASA-TLX subscales, n={len(tlx)} — individual points, not error bars")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "tlx.png", dpi=150)
    plt.close(fig)


def plot_time_on_task(tasks: pd.DataFrame, budget_s: float, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(tasks.task_id.unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, task in enumerate(order):
        sub = tasks[tasks.task_id == task]
        jitter = np.linspace(-0.12, 0.12, len(sub))
        colours = ["0.2" if s else "0.65" for s in sub.success]
        ax.scatter(np.full(len(sub), i) + jitter, sub.time_on_task_s,
                   c=colours, s=45, zorder=3, edgecolors="black", linewidths=0.5)
    ax.axhline(budget_s, linestyle="--", color="black",
               label=f"{budget_s:.0f} s budget")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("time on task (s)")
    ax.set_xlabel("task")
    ax.set_title("Time on task against the half-time budget (dark = success)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "time_on_task.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse pilot sessions.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tlx", default="pilot/tlx_responses.csv")
    ap.add_argument("--themes", default="pilot/themes.csv",
                    help="optional qualitative coding: theme,participant,quote_paraphrase")
    args = ap.parse_args()

    cfg = load_config(args.config)
    budget_s = float(cfg.dot("pilot.task_budget_s", 90))
    out_dir = cfg.get_path("paths.outputs_dir")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    events = load_pilot_events(cfg)
    if events.empty:
        raise SystemExit(
            "No pilot events in the database. Run sessions through the UI with the "
            "pilot panel enabled, or import session logs first."
        )

    tasks = summarise_tasks(events, budget_s)
    if tasks.empty:
        raise SystemExit("Pilot events found but no complete task_start/task_end pairs.")

    tasks.to_csv(out_dir / "pilot_tasks.csv", index=False)

    n_participants = tasks.participant.nunique()
    overall = {
        "n_participants": int(n_participants),
        "n_task_attempts": int(len(tasks)),
        "task_success_rate": round(float(tasks.success.mean()), 3),
        "within_budget_rate": round(float(tasks.within_budget.mean()), 3),
        "time_on_task_s": {
            "median": round(float(tasks.time_on_task_s.median()), 1),
            "iqr": [round(float(tasks.time_on_task_s.quantile(0.25)), 1),
                    round(float(tasks.time_on_task_s.quantile(0.75)), 1)],
            "min": round(float(tasks.time_on_task_s.min()), 1),
            "max": round(float(tasks.time_on_task_s.max()), 1),
        },
        "queries_per_task": {
            "median": float(tasks.n_queries.median()),
            "max": int(tasks.n_queries.max()),
        },
        "errors_total": int(tasks.n_errors.sum()),
    }

    by_task = (
        tasks.groupby("task_id")
        .agg(n=("success", "size"),
             success_rate=("success", "mean"),
             median_time_s=("time_on_task_s", "median"),
             median_queries=("n_queries", "median"),
             within_budget_rate=("within_budget", "mean"))
        .round(3).reset_index().to_dict(orient="records")
    )

    tlx_path = Path(args.tlx)
    if not tlx_path.is_absolute():
        tlx_path = REPO_ROOT / tlx_path
    tlx = load_tlx(tlx_path)
    tlx_summary = {}
    if not tlx.empty:
        tlx_summary = {
            "n": int(len(tlx)),
            "rtlx_mean": round(float(tlx.rtlx.mean()), 1),
            "rtlx_range": [round(float(tlx.rtlx.min()), 1), round(float(tlx.rtlx.max()), 1)],
            "subscales": {c: {"mean": round(float(tlx[c].mean()), 1),
                              "min": float(tlx[c].min()), "max": float(tlx[c].max())}
                          for c in TLX_SUBSCALES},
            "per_participant": tlx[["participant"] + TLX_SUBSCALES + ["rtlx"]]
            .to_dict(orient="records") if "participant" in tlx.columns else [],
        }
        plot_tlx(tlx, fig_dir)

    themes_path = Path(args.themes)
    if not themes_path.is_absolute():
        themes_path = REPO_ROOT / themes_path
    themes = {}
    if themes_path.exists():
        tdf = pd.read_csv(themes_path)
        if "theme" in tdf.columns:
            themes = {
                "n_coded_observations": int(len(tdf)),
                "themes": tdf.groupby("theme").size().sort_values(ascending=False).to_dict(),
            }

    plot_time_on_task(tasks, budget_s, fig_dir)

    results = {
        "budget_s": budget_s,
        "caveat": (
            "Descriptive only. n is too small for inferential statistics, and "
            "participants were football-literate testers rather than practising "
            "performance analysts."
        ),
        "overall": overall,
        "by_task": by_task,
        "nasa_tlx": tlx_summary,
        "think_aloud_themes": themes,
    }
    out = write_json(out_dir / "pilot_results.json", results)

    log.info("Participants: %d | attempts: %d | success rate: %.0f%%",
             overall["n_participants"], overall["n_task_attempts"],
             100 * overall["task_success_rate"])
    log.info("Median time on task: %.0f s (budget %.0f s); %.0f%% of attempts inside budget.",
             overall["time_on_task_s"]["median"], budget_s,
             100 * overall["within_budget_rate"])
    if tlx_summary:
        log.info("RTLX mean %.1f (range %.1f–%.1f).", tlx_summary["rtlx_mean"],
                 *tlx_summary["rtlx_range"])
    log.info("Wrote %s and figures.", out)


if __name__ == "__main__":
    main()
