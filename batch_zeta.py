"""Batch ZETA analysis: run the ZETA scripts for every recording session.

Two analyses are available:

  responsiveness  zeta_analysis.py — one-sample ZETA, tests whether each
                  neuron responds to each behavioural event (cue, response,
                  reward, trial_start).
                  Output: results/zeta_responsiveness/zeta_<event>_<session>.{csv,png}

  outcome         zeta_outcome.py  — two-sample ZETA, tests whether each
                  neuron's reward-aligned response differs between trial
                  outcomes (G+R vs G+N, G+R vs S+R).
                  Output: results/zeta_outcome/zeta2_<contrast>_<session>.{csv,png}

For each session it writes the full results tables (CSV) and the top-8
significant-neuron plots (PNG) to the appropriate results/ subfolder.

This wrapper just discovers sessions and invokes the underlying scripts with
`--session <id>` — it does NOT touch utils.py. Both scripts test neurons in
parallel across all CPU cores; pass --jobs to cap it.

Usage:
    python batch_zeta.py                          # both analyses, all sessions
    python batch_zeta.py --analysis outcome       # only the two-sample analysis
    python batch_zeta.py --analysis responsiveness
    python batch_zeta.py --sessions 20250521 20250602
    python batch_zeta.py --jobs 6 --resamp 100
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from utils import REPO_ROOT, RESULTS_DIR

HERE = Path(REPO_ROOT)

# Each analysis: the script to run and its fixed arguments. --csv + --save
# (no path) make each script auto-name its CSV/PNG per event/contrast and
# session; --top 8 plots the 8 most significant neurons.
ANALYSES = {
    "responsiveness": {
        "script": "zeta_analysis.py",
        "args":   ["--event", "all", "--csv", "--save", "--top", "8"],
    },
    "outcome": {
        "script": "zeta_outcome.py",
        "args":   ["--contrast", "all", "--csv", "--save", "--top", "8"],
    },
}


def discover_sessions():
    return sorted(
        d.name for d in HERE.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}', d.name)
    )


def parse_args():
    p = argparse.ArgumentParser(description="Batch ZETA analysis over all sessions.")
    p.add_argument("--analysis", default="both",
                   choices=list(ANALYSES.keys()) + ["both"],
                   help="Which analysis to run (default: both).")
    p.add_argument("--sessions", nargs="+", metavar="YYYYMMDD",
                   help="Sessions to process (default: every YYYYMMDD directory).")
    p.add_argument("--resamp", type=int,
                   help="Jitter iterations passed to the ZETA scripts (default: each script's own).")
    p.add_argument("--dur", type=float,
                   help="Analysis window in seconds passed to the ZETA scripts.")
    p.add_argument("--jobs", type=int,
                   help="Parallel worker processes per session (default: all CPU cores).")
    p.add_argument("--event", metavar="EVENT",
                   help="Restrict the responsiveness analysis to one event "
                        "(default: 'all'). Required when --window-end is set.")
    p.add_argument("--window-end", metavar="EVENT",
                   help="Variable-duration mode: per-trial [event, window_end]. "
                        "Forwarded to zeta_analysis.py (responsiveness only). "
                        "Implies and requires --event.")
    args = p.parse_args()
    if args.window_end and not args.event:
        p.error("--window-end requires --event (e.g. --event cue --window-end reward).")
    return args


def main():
    args     = parse_args()
    sessions = args.sessions or discover_sessions()
    if not sessions:
        print("No sessions found.")
        return

    selected = list(ANALYSES) if args.analysis == "both" else [args.analysis]

    # Optional passthrough flags applied to whichever script(s) run.
    extra = []
    if args.resamp is not None:
        extra += ["--resamp", str(args.resamp)]
    if args.dur is not None:
        extra += ["--dur", str(args.dur)]
    if args.jobs is not None:
        extra += ["--jobs", str(args.jobs)]

    # Per-analysis flags that override the script's fixed args in ANALYSES.
    # --event/--window-end only apply to the responsiveness script — the
    # outcome script has its own --contrast loop instead.
    responsiveness_overrides = []
    if args.event is not None:
        # Replace the default "--event all" with the user's pick.
        responsiveness_overrides += ["--event", args.event]
    if args.window_end is not None:
        responsiveness_overrides += ["--window-end", args.window_end]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Sessions: {sessions}")
    print(f"Analyses: {selected}\n")

    # Headless backend so plt.show() in the ZETA scripts never blocks the batch.
    env = dict(os.environ, MPLBACKEND="Agg")

    failed = []
    for session in sessions:
        print(f"=== {session} ===", flush=True)
        for name in selected:
            cfg = ANALYSES[name]
            # When the user passed --event / --window-end, swap out the
            # default "--event all" for the explicit selection.
            script_args = list(cfg["args"])
            if name == "responsiveness" and responsiveness_overrides:
                # Drop the default "--event all" pair so our override wins.
                if "--event" in script_args:
                    i = script_args.index("--event")
                    del script_args[i:i+2]
                script_args = responsiveness_overrides + script_args
            print(f"  -- {name}: {cfg['script']} {' '.join(script_args)} --", flush=True)
            result = subprocess.run(
                [sys.executable, str(HERE / cfg["script"]),
                 "--session", session, *script_args, *extra],
                env=env,
            )
            if result.returncode != 0:
                print(f"  WARNING: {cfg['script']} exited with code "
                      f"{result.returncode} for {session}")
                failed.append(f"{session}/{name}")
        print()

    n_runs = len(sessions) * len(selected)
    print(f"\nDone: {n_runs - len(failed)}/{n_runs} runs -> CSVs and plots in {RESULTS_DIR}")
    if failed:
        print(f"Failed runs: {failed}")


if __name__ == "__main__":
    main()
