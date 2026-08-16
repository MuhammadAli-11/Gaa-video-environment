"""make_known_item_queries.py — build an evaluation set without exhaustive judging.

Full graded relevance means judging every clip against every query: ~20 queries x
~20 clips. A known-item design instead asks you to describe each clip once, and
treats that description as a query whose single correct answer is that clip. The
labelling falls out of watching the footage, which you have to do anyway.

This is a standard IR evaluation design, not a shortcut invented for a deadline, but
it measures something narrower and the write-up must say which was used. See the
caveats printed at the end.

Two steps:

    python tools/make_known_item_queries.py --init
        writes eval/descriptions.yaml, one blank entry per clip

    # ...watch the clips, fill in a sentence for each...

    python tools/make_known_item_queries.py
        converts those into eval/queries.yaml, ready for s05 and s07
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import REPO_ROOT, connect, load_config, setup_logging  # noqa: E402

log = setup_logging("known_item")

HOWTO = """\
Write one sentence per clip describing what a person would SEE in it.

  good:  "keeper kicks long down the middle, two players jump for it"
  good:  "short kickout to a defender on the left, no contest"
  bad:   "kickout 4"                 (no visual content to match on)
  bad:   "the one Galway won"        (needs knowledge the model cannot have)

Three rules that keep this honest:

  1. Describe what is on screen, not what you know from the scoreboard.
  2. Write every description BEFORE you run any search. Descriptions written after
     seeing results are tuned to the system, and the evaluation stops measuring
     retrieval and starts measuring your memory of the output.
  3. Leave `skip: true` on replays and unusable clips. A second angle on a clip you
     already described makes the "single correct answer" assumption false.
"""


def init_template(cfg, path: Path) -> None:
    conn = connect(cfg)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.clip_id, e.event_type, e.t_peak_s, e.pitch_zone
            FROM clips c JOIN events e ON e.event_id = c.event_id
            ORDER BY e.t_peak_s
        """)
        rows = cur.fetchall()
    conn.close()
    if not rows:
        raise SystemExit("No clips found. Run s01_extract_clips.py first.")

    doc = {
        "_howto": HOWTO,
        "clips": [
            {
                "clip_id": r[0],
                "at": f"{int((r[2] or 0)//60):02d}:{int((r[2] or 0)%60):02d}",
                "event_type": r[1],
                "pitch_zone": r[3],
                "description": "",
                "skip": False,
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True, width=100)
    log.info("Wrote %s with %d clips.", path, len(rows))
    log.info("Fill in every `description`, mark replays `skip: true`, then re-run.")


def convert(desc_path: Path, out_path: Path, use_filters: bool) -> None:
    doc = yaml.safe_load(desc_path.read_text())
    clips = doc.get("clips", [])
    usable = [c for c in clips
              if str(c.get("description", "")).strip() and not c.get("skip")]
    if len(usable) < 8:
        raise SystemExit(
            f"Only {len(usable)} clips have descriptions. Fill in at least 8 — "
            "below that the confidence interval is too wide to say anything."
        )

    queries = []
    for i, c in enumerate(usable, 1):
        q = {
            "id": f"k{i:02d}",
            "text": str(c["description"]).strip(),
            "relevant": {int(c["clip_id"]): 2},
        }
        # Filters drive the hybrid arm. Only include the ones the analyst could
        # plausibly have set from the query itself, and never derive them from the
        # target clip's own metadata — that hands hybrid the answer and turns the
        # comparison into a tautology.
        if use_filters and c.get("event_type"):
            q["filters"] = {"event_type": c["event_type"]}
        queries.append(q)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(
            {"_design": "known-item: exactly one relevant clip per query",
             "queries": queries},
            fh, sort_keys=False, allow_unicode=True, width=100,
        )

    skipped = len(clips) - len(usable)
    log.info("Wrote %s: %d known-item queries (%d clips skipped or blank).",
             out_path, len(queries), skipped)
    log.info("")
    log.info("Next:")
    log.info("    python src/s05_benchmark.py --mode retrieval")
    log.info("    python src/s07_stats.py --primary RR")
    log.info("")
    log.warning("Report these as known-item results, and state three things:")
    log.warning("  * With one relevant clip per query, P@1 is success@1 and R@10 is")
    log.warning("    success@10. P@3 and P@5 are capped at 0.33 and 0.20 and mean")
    log.warning("    nothing here — do not report them. MRR is the headline.")
    log.warning("  * Known-item flatters the system: the query is a faithful")
    log.warning("    description of the target, whereas a real analyst is guessing at")
    log.warning("    vocabulary. Treat the numbers as an upper bound.")
    log.warning("  * Zone filters were NOT auto-derived from the target clip. Deriving")
    log.warning("    them would guarantee the hybrid win and prove nothing.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a known-item evaluation set.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--init", action="store_true", help="write the blank description template")
    ap.add_argument("--descriptions", default="eval/descriptions.yaml")
    ap.add_argument("--out", default="eval/queries.yaml")
    ap.add_argument("--no-filters", action="store_true",
                    help="omit event_type filters, making both arms identical")
    args = ap.parse_args()

    cfg = load_config(args.config)
    desc_path = Path(args.descriptions)
    out_path = Path(args.out)
    if not desc_path.is_absolute():
        desc_path = REPO_ROOT / desc_path
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    if args.init:
        init_template(cfg, desc_path)
        return
    if not desc_path.exists():
        raise SystemExit(f"{desc_path} not found. Run with --init first.")
    convert(desc_path, out_path, use_filters=not args.no_filters)


if __name__ == "__main__":
    main()
