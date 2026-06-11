"""Firing-rate vs HGF prediction-error (δ₁) grid CLI — one subplot per neuron.

Same plot family as `firing_rate_vs_perc_p.py` / `firing_rate_vs_hgf_p.py`, but
the x-axis is the HGF level-1 prediction error δ₁ = outcome − p̂ per trial, read
from `results/hgf/trajectory_<session>.csv`.

δ₁ is a genuine reward prediction error only when a gamble outcome is observed,
so only gamble trials are shown (on safe trials δ₁ collapses to −p̂). By default
points are coloured by outcome (G+R = positive PE, G+N = negative PE) with an
independent regression per condition; pass `--pooled` for a single global fit.

Run `python -m analysis.hgf.run` first to generate the trajectory CSVs.

Thin wrapper over `explore.grid(sess, "fr_vs_delta1", ...)`. Single neuron:
    Session(...).neuron(3).fr_vs_delta1(window="cue_to_reward").save()

    python -m viewers.firing_rate_vs_delta1 --window cue_to_reward --bins 8
    python -m viewers.firing_rate_vs_delta1 --neurons 0 1 5 --area ACC --pooled --save
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib.pyplot as plt

from session import Session
from explore import grid
from compute import WINDOWS
from utils import (
    add_session_arg, add_selection_args, add_save_arg, maybe_save,
    handle_list, session_data_dir,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Firing rate vs HGF prediction error (delta1) grid")
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument("--window", default="trial", choices=list(WINDOWS),
                        help="Time window for firing rate (default: trial)")
    parser.add_argument("--bins", type=int, default=8, metavar="N",
                        help="Number of bins for the pooled mean overlay (default: 8)")
    parser.add_argument("--pooled", action="store_true",
                        help="Single global fit instead of one regression per outcome condition")
    add_save_arg(parser)
    args = parser.parse_args()

    if handle_list(args, data_dir=session_data_dir(args.session)):
        raise SystemExit

    sess  = Session(args.session)
    panel = grid(sess, "fr_vs_delta1", neurons=args.neurons, area=args.area,
                 window=args.window, n_bins=args.bins, by_condition=not args.pooled)
    maybe_save(panel.fig, args, prefix="fr_vs_delta1")
    plt.show()
