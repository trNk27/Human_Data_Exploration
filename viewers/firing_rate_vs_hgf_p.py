"""Firing-rate vs HGF perceived gamble probability grid CLI — one subplot per neuron.

Like `firing_rate_vs_perc_p.py`, but the x-axis is the HGF model's *latent*
perceived gamble-reward probability (p̂) per trial — read from
`results/hgf/trajectory_<session>.csv` — instead of the behavioural rolling
reward rate. Per subplot: per-trial scatter, binned mean ± SEM, and a linear fit.

Because the HGF belief is defined on every responding trial (not just gamble
trials), all responding trials are used. Run `python -m analysis.hgf.run` first
to generate the trajectory CSVs.

Thin wrapper over `explore.grid(sess, "fr_vs_hgf_p", ...)`. Single neuron:
    Session(...).neuron(3).fr_vs_hgf_p(window="cue_to_reward").save()

    python -m viewers.firing_rate_vs_hgf_p --window cue_to_reward --bins 8
    python -m viewers.firing_rate_vs_hgf_p --neurons 0 1 5 --area ACC --save
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
    parser = argparse.ArgumentParser(description="Firing rate vs HGF perceived gamble probability grid")
    add_session_arg(parser)
    add_selection_args(parser)
    parser.add_argument("--window", default="trial", choices=list(WINDOWS),
                        help="Time window for firing rate (default: trial)")
    parser.add_argument("--bins", type=int, default=8, metavar="N",
                        help="Number of probability bins for mean overlay (default: 8)")
    parser.add_argument("--by-condition", action="store_true",
                        help="Colour points by outcome and fit one regression per condition")
    add_save_arg(parser)
    args = parser.parse_args()

    if handle_list(args, data_dir=session_data_dir(args.session)):
        raise SystemExit

    sess  = Session(args.session)
    panel = grid(sess, "fr_vs_hgf_p", neurons=args.neurons, area=args.area,
                 window=args.window, n_bins=args.bins, by_condition=args.by_condition)
    maybe_save(panel.fig, args, prefix="fr_vs_hgf_p")
    plt.show()
