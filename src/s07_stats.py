"""s07_stats.py — statistical analysis of the evaluation.

Three separate inferential problems, with three different appropriate answers:

1. RETRIEVAL (n = number of queries). The semantic and hybrid arms are run on the
   *same* queries, so every comparison is paired and the per-query difference is the
   unit of analysis. Standard practice in IR evaluation is a paired randomisation
   (permutation) test rather than a t-test, because per-query metric distributions
   are bounded, discrete and badly non-normal — P@5 can only take six values.
   Reported with a bootstrap confidence interval on the mean difference, which is
   more informative than the p-value and survives contact with a reviewer better.

2. LATENCY (n = number of timed requests). Large n, so the interesting quantity is
   not a mean but a tail percentile, and a p95 estimated from a few hundred samples
   is noisier than people assume. Bootstrap CI on the percentile itself.

3. PILOT (n = 3–5 participants). No inferential statistics. This script computes the
   Wilson interval on the success rate anyway — not to report as a finding, but to
   show how wide it is. Demonstrating that the interval spans most of the unit
   interval is a stronger argument for descriptive-only reporting than asserting it.

Multiplicity: several metrics are compared on the same data, so a primary metric is
pre-specified (nDCG@10, the only rank-weighted graded measure among them) and
Holm-Bonferroni adjusted p-values are reported for the family of secondary metrics.

Usage:
    python src/s07_stats.py
    python src/s07_stats.py --primary "P@5" --bootstrap 20000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import load_config, setup_logging, write_json

log = setup_logging("s07_stats")

METRICS = ["P@1", "P@3", "P@5", "R@10", "RR", "nDCG@10"]
EXACT_PERM_LIMIT = 16      # 2^16 = 65,536 sign patterns: enumerate exactly below this


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 statistic=np.mean, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI. Resamples the queries, not the clips — the query set
    is the sample, and treating individual judgements as independent would overstate
    precision considerably."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = statistic(values[idx], axis=1)
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def paired_permutation_test(diff: np.ndarray, n_perm: int = 100_000,
                            seed: int = 0) -> tuple[float, bool]:
    """Two-sided paired randomisation test on the mean difference.

    Null hypothesis: the arm labels are exchangeable within each query, so each
    observed difference is equally likely to have carried the opposite sign. Exact
    enumeration of all 2^n sign patterns when n is small enough, Monte Carlo above
    that. Returns (p_value, was_exact).

    Zero differences are retained rather than dropped: under this null a tie is
    evidence of no effect and discarding ties inflates the apparent signal.
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n == 0 or np.allclose(diff, 0):
        return (1.0, True)
    observed = abs(diff.mean())

    if n <= EXACT_PERM_LIMIT:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        means = np.abs((signs * diff).mean(axis=1))
        # >= with a tolerance: floating-point equality would otherwise miss the
        # observed arrangement itself, which must always be counted.
        return (float((means >= observed - 1e-12).mean()), True)

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    means = np.abs((signs * diff).mean(axis=1))
    # (count + 1) / (n_perm + 1): the observed arrangement is a member of the null
    # distribution, and this keeps p strictly positive.
    return (float((np.sum(means >= observed - 1e-12) + 1) / (n_perm + 1)), False)


def wilcoxon_signed_rank(diff: np.ndarray) -> dict:
    """Secondary test. scipy if present, otherwise reported as unavailable.

    Included because reviewers expect it, but the permutation test is the primary:
    Wilcoxon discards zero differences and P@k produces a lot of them at small k,
    which costs real power on a query set this size.
    """
    nonzero = int(np.sum(np.asarray(diff) != 0))
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"available": False, "n_nonzero": nonzero,
                "note": "pip install scipy to enable"}
    if nonzero < 1:
        return {"available": False, "n_nonzero": 0, "note": "all differences are zero"}
    stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return {"available": True, "statistic": float(stat), "p_value": float(p),
            "n_nonzero": nonzero}


def cohens_dz(diff: np.ndarray) -> float:
    """Paired effect size: mean difference in units of the SD of the differences."""
    sd = np.std(diff, ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("inf" if diff.mean() else 0.0)


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """Step-down adjustment. Controls family-wise error without the raw
    Bonferroni's power cost."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min((m - i) * p, 1.0))
        adjusted[key] = round(running, 6)
    return adjusted


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Correct near 0 and 1, where the normal approximation
    produces intervals that run outside [0, 1] and embarrass everyone."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def analyse_retrieval(evals: dict, primary: str, n_boot: int, n_perm: int) -> dict:
    per_query = evals.get("per_query", [])
    if len(per_query) < 3:
        raise SystemExit(
            f"Only {len(per_query)} judged queries. Statistical comparison needs a "
            "real query set — aim for 15–20 (see eval/queries.example.yaml)."
        )

    n = len(per_query)
    out = {
        "n_queries": n,
        "primary_metric": primary,
        "unit_of_analysis": "query (paired: both arms run on the same query set)",
        "metrics": {},
    }
    raw_p = {}

    for metric in METRICS:
        sem = np.array([q["semantic"].get(metric) for q in per_query], dtype=float)
        hyb = np.array([q["hybrid"].get(metric) for q in per_query], dtype=float)
        keep = ~(np.isnan(sem) | np.isnan(hyb))
        sem, hyb = sem[keep], hyb[keep]
        if len(sem) < 3:
            continue
        diff = hyb - sem

        p_perm, exact = paired_permutation_test(diff, n_perm=n_perm)
        raw_p[metric] = p_perm

        out["metrics"][metric] = {
            "semantic": {
                "mean": round(float(sem.mean()), 4),
                "sd": round(float(sem.std(ddof=1)), 4),
                "ci95": [round(v, 4) for v in bootstrap_ci(sem, n_boot)],
            },
            "hybrid": {
                "mean": round(float(hyb.mean()), 4),
                "sd": round(float(hyb.std(ddof=1)), 4),
                "ci95": [round(v, 4) for v in bootstrap_ci(hyb, n_boot)],
            },
            "difference": {
                "mean": round(float(diff.mean()), 4),
                "ci95": [round(v, 4) for v in bootstrap_ci(diff, n_boot)],
                "relative_pct": round(100 * diff.mean() / sem.mean(), 1) if sem.mean() else None,
            },
            "permutation_test": {"p_value": round(p_perm, 5), "exact": exact,
                                 "n_permutations": 2 ** len(diff) if exact else n_perm},
            "wilcoxon": wilcoxon_signed_rank(diff),
            "effect_size_dz": round(cohens_dz(diff), 3),
            "wins_losses_ties": {
                "hybrid_better": int(np.sum(diff > 0)),
                "semantic_better": int(np.sum(diff < 0)),
                "tied": int(np.sum(diff == 0)),
            },
        }

    secondary = {k: v for k, v in raw_p.items() if k != primary}
    out["holm_adjusted_p_secondary"] = holm_bonferroni(secondary) if secondary else {}
    out["multiplicity_note"] = (
        f"{primary} pre-specified as primary and reported unadjusted. The remaining "
        f"{len(secondary)} metrics are a family and carry Holm-Bonferroni adjusted "
        "p-values; they are secondary evidence, not independent confirmations."
    )
    return out


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def analyse_latency(bench: dict, n_boot: int) -> dict:
    raw = np.array(bench.get("warm", {}).get("raw_e2e_ms", []), dtype=float)
    if raw.size < 10:
        return {"available": False, "note": "not enough timed requests in benchmark.json"}

    def pct_ci(p):
        lo, hi = bootstrap_ci(raw, n_boot, statistic=lambda a, axis: np.percentile(a, p, axis=axis))
        return [round(lo, 1), round(hi, 1)]

    p95 = float(np.percentile(raw, 95))
    budget_ms = 500.0
    return {
        "available": True,
        "n_requests": int(raw.size),
        "p50_ms": round(float(np.percentile(raw, 50)), 1),
        "p50_ci95": pct_ci(50),
        "p95_ms": round(p95, 1),
        "p95_ci95": pct_ci(95),
        "p99_ms": round(float(np.percentile(raw, 99)), 1),
        "mean_ms": round(float(raw.mean()), 1),
        "sd_ms": round(float(raw.std(ddof=1)), 1),
        "proportion_under_500ms": round(float((raw < budget_ms).mean()), 4),
        "proportion_under_500ms_ci95": [
            round(v, 4) for v in wilson_interval(int((raw < budget_ms).sum()), raw.size)
        ],
        "note": (
            "A p95 from a few hundred requests is an estimate with real width; the CI "
            "is what should be quoted, not the point value. Reported against the "
            "500 ms design target rather than tested — a one-sided test against a "
            "target the system was built to hit answers a question nobody asked."
        ),
    }


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------
def analyse_pilot(pilot: dict) -> dict:
    overall = pilot.get("overall", {})
    n_attempts = overall.get("n_task_attempts", 0)
    n_participants = overall.get("n_participants", 0)
    if not n_attempts:
        return {"available": False}

    rate = overall.get("task_success_rate", 0.0)
    successes = int(round(rate * n_attempts))
    lo, hi = wilson_interval(successes, n_attempts)

    within = overall.get("within_budget_rate", 0.0)
    w_lo, w_hi = wilson_interval(int(round(within * n_attempts)), n_attempts)

    return {
        "available": True,
        "n_participants": n_participants,
        "n_attempts": n_attempts,
        "task_success_rate": rate,
        "task_success_ci95_wilson": [round(lo, 3), round(hi, 3)],
        "ci_width": round(hi - lo, 3),
        "within_budget_rate": within,
        "within_budget_ci95_wilson": [round(w_lo, 3), round(w_hi, 3)],
        "inference_performed": False,
        "interpretation": (
            f"The 95% interval on success rate spans {round(hi - lo, 2)} of the unit "
            f"interval at n={n_attempts} attempts from {n_participants} participants. "
            "It is reported to show why this pilot is analysed descriptively, not as a "
            "finding. Attempts are also not independent — each participant contributes "
            "several, and they learn the interface as they go — so even this interval "
            "is optimistic. No hypothesis test is run on the pilot data."
        ),
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def plot_paired(evals: dict, primary: str, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_query = evals["per_query"]
    sem = np.array([q["semantic"].get(primary, 0) for q in per_query], dtype=float)
    hyb = np.array([q["hybrid"].get(primary, 0) for q in per_query], dtype=float)
    order = np.argsort(sem)
    sem, hyb = sem[order], hyb[order]
    y = np.arange(len(sem))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # Paired slope plot: every query visible, no aggregation hiding the variance.
    for i in range(len(sem)):
        axes[0].plot([sem[i], hyb[i]], [y[i], y[i]],
                     color="0.3" if hyb[i] >= sem[i] else "0.65",
                     linewidth=1.2, zorder=1)
    axes[0].scatter(sem, y, s=34, facecolor="white", edgecolor="black", zorder=2, label="semantic")
    axes[0].scatter(hyb, y, s=34, color="black", zorder=3, label="hybrid")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([per_query[i].get("id", f"q{i}") for i in order], fontsize=7)
    axes[0].set_xlabel(primary)
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_title(f"Per-query {primary}, paired")
    axes[0].legend(fontsize=8, loc="lower right")

    diff = hyb - sem
    lo, hi = bootstrap_ci(diff)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].hist(diff, bins=max(6, len(diff) // 2), color="0.75", edgecolor="black")
    axes[1].axvline(diff.mean(), color="black", linestyle="-", linewidth=2,
                    label=f"mean {diff.mean():+.3f}")
    axes[1].axvspan(lo, hi, color="black", alpha=0.12, label=f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    axes[1].set_xlabel(f"hybrid − semantic ({primary})")
    axes[1].set_ylabel("queries")
    axes[1].set_title("Paired differences")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "paired_retrieval.png", dpi=150)
    plt.close(fig)


def write_markdown(results: dict, path: Path) -> Path:
    r = results.get("retrieval", {})
    lat = results.get("latency", {})
    pil = results.get("pilot", {})
    primary = r.get("primary_metric")
    lines = [
        "# Statistical analysis",
        "",
        f"Generated by `src/s07_stats.py`. Query set n={r.get('n_queries', '—')}; "
        f"primary metric **{primary}**, pre-specified.",
        "",
        "## Retrieval: hybrid versus semantic",
        "",
        "Paired comparison — both arms run on the same queries, so the per-query "
        "difference is the unit of analysis. Two-sided paired randomisation test; "
        "95% CIs from a percentile bootstrap over queries.",
        "",
        "| Metric | Semantic (95% CI) | Hybrid (95% CI) | Difference (95% CI) | p | d_z | W/L/T |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in r.get("metrics", {}).items():
        s, h, d = m["semantic"], m["hybrid"], m["difference"]
        wlt = m["wins_losses_ties"]
        star = " *" if name == primary else ""
        lines.append(
            f"| {name}{star} | {s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] "
            f"| {h['mean']:.3f} [{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}] "
            f"| {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
            f"| {m['permutation_test']['p_value']:.4f} | {m['effect_size_dz']:+.2f} "
            f"| {wlt['hybrid_better']}/{wlt['semantic_better']}/{wlt['tied']} |"
        )
    lines += ["", f"\\* primary metric, unadjusted. {r.get('multiplicity_note', '')}", ""]

    if r.get("holm_adjusted_p_secondary"):
        lines += ["Holm-adjusted p-values for the secondary family: "
                  + ", ".join(f"{k} = {v:.4f}" for k, v in r["holm_adjusted_p_secondary"].items()),
                  ""]

    if lat.get("available"):
        lines += [
            "## Latency",
            "",
            f"n = {lat['n_requests']} timed warm requests.",
            "",
            "| Statistic | Value | 95% CI |",
            "|---|---|---|",
            f"| p50 | {lat['p50_ms']} ms | [{lat['p50_ci95'][0]}, {lat['p50_ci95'][1]}] |",
            f"| p95 | {lat['p95_ms']} ms | [{lat['p95_ci95'][0]}, {lat['p95_ci95'][1]}] |",
            f"| p99 | {lat['p99_ms']} ms | — |",
            f"| Under 500 ms | {lat['proportion_under_500ms']:.1%} | "
            f"[{lat['proportion_under_500ms_ci95'][0]:.1%}, "
            f"{lat['proportion_under_500ms_ci95'][1]:.1%}] |",
            "",
            lat["note"],
            "",
        ]

    if pil.get("available"):
        lines += [
            "## Pilot",
            "",
            f"n = {pil['n_participants']} participants, {pil['n_attempts']} task attempts. "
            "**No hypothesis test is performed.**",
            "",
            f"- Task success rate {pil['task_success_rate']:.0%}, Wilson 95% CI "
            f"[{pil['task_success_ci95_wilson'][0]:.0%}, {pil['task_success_ci95_wilson'][1]:.0%}]",
            f"- Within budget {pil['within_budget_rate']:.0%}, Wilson 95% CI "
            f"[{pil['within_budget_ci95_wilson'][0]:.0%}, {pil['within_budget_ci95_wilson'][1]:.0%}]",
            "",
            pil["interpretation"],
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Statistical analysis of the evaluation.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--primary", default="nDCG@10", choices=METRICS)
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--permutations", type=int, default=100_000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get_path("paths.outputs_dir")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    eval_path = out_dir / "retrieval_eval.json"
    if not eval_path.exists():
        raise SystemExit(
            f"{eval_path} not found. Run the retrieval evaluation first:\n"
            "    python src/s05_benchmark.py --mode retrieval"
        )
    evals = json.loads(eval_path.read_text())

    results = {
        "retrieval": analyse_retrieval(evals, args.primary, args.bootstrap, args.permutations),
    }

    bench_path = out_dir / "benchmark.json"
    if bench_path.exists():
        results["latency"] = analyse_latency(json.loads(bench_path.read_text()), args.bootstrap)

    pilot_path = out_dir / "pilot_results.json"
    if pilot_path.exists():
        results["pilot"] = analyse_pilot(json.loads(pilot_path.read_text()))

    plot_paired(evals, args.primary, fig_dir)
    write_json(out_dir / "statistical_analysis.json", results)
    md = write_markdown(results, out_dir / "statistical_report.md")

    prim = results["retrieval"]["metrics"].get(args.primary)
    if prim:
        d = prim["difference"]
        log.info("%s: semantic %.3f → hybrid %.3f (%+.3f, 95%% CI [%+.3f, %+.3f]), p = %.4f, d_z = %+.2f",
                 args.primary, prim["semantic"]["mean"], prim["hybrid"]["mean"],
                 d["mean"], d["ci95"][0], d["ci95"][1],
                 prim["permutation_test"]["p_value"], prim["effect_size_dz"])
        w = prim["wins_losses_ties"]
        log.info("Hybrid better on %d queries, worse on %d, tied on %d.",
                 w["hybrid_better"], w["semantic_better"], w["tied"])
    log.info("Wrote %s", md)


if __name__ == "__main__":
    main()
