"""Run zeta_analysis.py for every session and every event.

For each session it runs the ZETA test against all events (cue, response,
reward, trial_start), writes the full results table per event to results/
as a CSV, and saves the IFR grid plot of the 8 most significant neurons per
event as a PNG (also in results/).

Mirrors the session-sweep approach of batch_export_acg.py: it temporarily
rewrites SESSION in utils.py per session and restores it afterwards.

Note: the ZETA test is slow (~15-20 s per neuron at the default 100
resamples). A full 8-session sweep can take many hours — use --resamp to
trade statistical resolution for speed.

Usage:
    python batch_zeta.py
    python batch_zeta.py --sessions 20250521 20250602
    python batch_zeta.py --resamp 50 --dur 1.5
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

# zeta_analysis.py with --save (no path) auto-names plots zeta_<event>_<SESSION>.png
# in the working directory; --csv writes zeta_<event>_<SESSION>.csv to results/.
# --event all + --top 8 give every event and the 8 most significant neurons.
BASE_ZETA_ARGS = ["--event", "all", "--csv", "--save", "--top", "8"]


def set_session(text: str, session: str) -> str:
    return SESSION_RE.sub(rf'\g<1>{session}\g<2>', text)


def discover_sessions():
    return sorted(
        d.name for d in HERE.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}', d.name)
    )


def parse_args():
    p = argparse.ArgumentParser(description="Batch ZETA analysis over all sessions.")
    p.add_argument("--sessions", nargs="+", metavar="YYYYMMDD",
                   help="Sessions to process (default: every YYYYMMDD directory).")
    p.add_argument("--resamp", type=int,
                   help="Jitter iterations passed to zeta_analysis.py (default: its own).")
    p.add_argument("--dur", type=float,
                   help="Analysis window in seconds passed to zeta_analysis.py.")
    return p.parse_args()


def main():
    args     = parse_args()
    sessions = args.sessions or discover_sessions()
    if not sessions:
        print("No sessions found.")
        return

    zeta_args = list(BASE_ZETA_ARGS)
    if args.resamp is not None:
        zeta_args += ["--resamp", str(args.resamp)]
    if args.dur is not None:
        zeta_args += ["--dur", str(args.dur)]

    RESULTS.mkdir(exist_ok=True)
    print(f"Sessions to process: {sessions}\n")

    # Run with a headless backend so plt.show() in zeta_analysis.py never blocks.
    env = dict(os.environ, MPLBACKEND="Agg")

    original = UTILS.read_text(encoding="utf-8")
    failed   = []
    try:
        for session in sessions:
            print(f"=== {session} ===", flush=True)
            UTILS.write_text(set_session(original, session), encoding="utf-8")
            # cwd=results/ so the auto-named PNGs land there alongside the CSVs.
            result = subprocess.run(
                [sys.executable, str(HERE / "zeta_analysis.py"), *zeta_args],
                cwd=str(RESULTS),
                env=env,
            )
            if result.returncode != 0:
                print(f"  WARNING: zeta_analysis.py exited with code {result.returncode} for {session}")
                failed.append(session)
            print()
    finally:
        UTILS.write_text(original, encoding="utf-8")
        print("utils.py restored.")

    done = [s for s in sessions if s not in failed]
    print(f"\nDone: {len(done)}/{len(sessions)} sessions → CSVs and plots in {RESULTS}")
    if failed:
        print(f"Failed sessions: {failed}")


if __name__ == "__main__":
    main()
