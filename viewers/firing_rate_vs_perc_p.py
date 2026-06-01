"""Firing-rate vs perceived-P(reward) grid CLI — one subplot per neuron.

Perceived probability at trial t is the rolling fraction of rewards in the last
`history` gamble trials before t. Per subplot: per-trial scatter, binned
mean ± SEM, and a linear fit (gamble trials only).

Thin wrapper over `explore.grid(sess, "fr_vs_p", ...)`. Single neuron:
    Session(...).neuron(3).fr_vs_p(window="cue_to_reward").save()

The computations live in `compute` (perceived_probability, trial_firing_rates).

    python -m viewers.firing_rate_vs_perc_p --window cue_to_reward --history 10 --bins 8
    python -m viewers.firing_rate_vs_perc_p --neurons 0 1 5 --area ACC --save
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
    parser = argparse.ArgumentParser(description="Firing rate vs perceived P(reward) grid")
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument("--window", default="trial", choices=list(WINDOWS),
                        help="Time window for firing rate (default: trial)")
    parser.add_argument("--history", type=int, default=10, metavar="N",
                        help="Past-trial window for perceived probability (default: 10)")
    parser.add_argument("--bins", type=int, default=8, metavar="N",
                        help="Number of probability bins for mean overlay (default: 8)")
    add_save_arg(parser)
    args = parser.parse_args()

    if handle_list(args, data_dir=session_data_dir(args.session)):
        raise SystemExit

    sess  = Session(args.session)
    panel = grid(sess, "fr_vs_p", neurons=args.neurons, area=args.area,
                 window=args.window, history=args.history, n_bins=args.bins)
    maybe_save(panel.fig, args, prefix="fr_vs_perc_p")
    plt.show()
