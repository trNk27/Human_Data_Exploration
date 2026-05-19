"""Batch ZETA analysis: run the ZETA scripts for every recording session.

Two analyses are available:

  responsiveness  zeta_analysis.py — one-sample ZETA, tests whether each
                  neuron responds to each behavioural event (cue, response,
                  reward, trial_start).
                  Output: zeta_<event>_<session>.{csv,png}

  outcome         zeta_outcome.py  — two-sample ZETA, tests whether each
                  neuron's reward-aligned response differs between trial
                  outcomes (G+R vs G+N, G+R vs S+R).
                  Output: zeta2_<contrast>_<session>.{csv,png}

For each session it writes the full results tables (CSV) and the top-8
significant-neuron plots (PNG) to results/.

Mirrors the session-sweep approach of batch_export_acg.py: it temporarily
rewrites SESSION in utils.py per session and restores it afterwards. Both
scripts test neurons in parallel across all CPU cores; pass --jobs to cap it.

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

HERE       = Path(__file__).parent
UTILS      = HERE / "utils.py"
RESULTS    = HERE / "results"
SESSION_RE = re.compile(r'^(SESSION\s*=\s*")[^"]*(")', re.MULTILINE)

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


def set_session(text: str, session: str) -> str:
    return SESSION_RE.sub(rf'\g<1>{session}\g<2>', text)


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
    return p.parse_args()


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

    RESULTS.mkdir(exist_ok=True)
    print(f"Sessions: {sessions}")
    print(f"Analyses: {selected}\n")

    # Headless backend so plt.show() in the ZETA scripts never blocks the batch.
    env = dict(os.environ, MPLBACKEND="Agg")

    original = UTILS.read_text(encoding="utf-8")
    failed   = []
    try:
        for session in sessions:
            print(f"=== {session} ===", flush=True)
            UTILS.write_text(set_session(original, session), encoding="utf-8")
            for name in selected:
                cfg = ANALYSES[name]
                print(f"  -- {name}: {cfg['script']} --", flush=True)
                # cwd=results/ so the auto-named PNGs land there beside the CSVs.
                result = subprocess.run(
                    [sys.executable, str(HERE / cfg["script"]), *cfg["args"], *extra],
                    cwd=str(RESULTS),
                    env=env,
                )
                if result.returncode != 0:
                    print(f"  WARNING: {cfg['script']} exited with code "
                          f"{result.returncode} for {session}")
                    failed.append(f"{session}/{name}")
            print()
    finally:
        UTILS.write_text(original, encoding="utf-8")
        print("utils.py restored.")

    n_runs = len(sessions) * len(selected)
    print(f"\nDone: {n_runs - len(failed)}/{n_runs} runs -> CSVs and plots in {RESULTS}")
    if failed:
        print(f"Failed runs: {failed}")


if __name__ == "__main__":
    main()
