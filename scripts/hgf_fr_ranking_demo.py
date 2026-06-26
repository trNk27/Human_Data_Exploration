"""Slide-18 figures: single-neuron firing rate vs HGF latent variables.

Two grids for one session, each showing the TOP-N neurons whose window firing
rate most strongly tracks an HGF latent variable (ranked by |Pearson r| across
trials, all regions pooled):

  1. fr_vs_delta1_top<N>_reward_to_end.png
        x = HGF prediction error  δ₁ = outcome − p̂   (gamble trials only)
        y = reward_to_end firing rate                (the PE is post-outcome)
        Coloured by outcome (G+R = positive PE, G+N = negative PE).

  2. fr_vs_hgf_p_top<N>_cue_to_reward.png
        x = HGF perceived gamble-reward probability p̂  (all responding trials)
        y = cue_to_reward firing rate                  (the belief is pre-outcome)

Ranking reuses the exact regressor + firing-rate plumbing the viewers use
(utils.load_hgf_*, compute.trial_firing_rates), so the |r| used to pick neurons
matches what the scatter shows. Reads results/hgf/trajectory_<session>.csv.

Usage:
    python scripts/hgf_fr_ranking_demo.py
    python scripts/hgf_fr_ranking_demo.py --session 20250521 --top 9
"""
import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from session import Session
from explore import grid
from compute import trial_firing_rates
from utils import (RESULTS_DIR, load_hgf_trajectory_column,
                   load_hgf_perceived_prob)

DEFAULT_SESSION = "20250521"
DEFAULT_TOP     = 9
MIN_PTS         = 20   # need enough finite trial pairs to trust a correlation


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default=DEFAULT_SESSION)
    p.add_argument("--top", type=int, default=DEFAULT_TOP,
                   help=f"Number of top-|r| neurons to show (default: {DEFAULT_TOP}).")
    return p.parse_args()


def rank_by_abs_r(x, rates, labels, top_n):
    """Return the top_n neuron indices by |Pearson r| between x and each column."""
    scored = []
    for j in range(rates.shape[1]):
        y  = rates[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < MIN_PTS:
            continue
        xv, yv = x[ok], y[ok]
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue
        r = float(np.corrcoef(xv, yv)[0, 1])
        scored.append((j, r, int(ok.sum())))
    scored.sort(key=lambda t: abs(t[1]), reverse=True)
    top = scored[:top_n]
    for rank, (j, r, n) in enumerate(top, 1):
        print(f"   {rank:2d}. neuron {j:3d}  r={r:+.3f}  (n={n})  {labels[j]}")
    return [j for j, _, _ in top]


def main():
    args = parse_args()
    sess = Session(args.session)
    trains, labels = sess.spike_trains
    trials, sr = sess.trials, sess.sampling_rate
    resp = sess.responding_mask
    arm  = trials["ChosenArm_G1S0"].to_numpy()
    gamble_mask = resp & (arm == 1)
    print(f"Session {args.session}: {len(trains)} neurons, "
          f"{int(resp.sum())} responding ({int(gamble_mask.sum())} gamble) trials")

    out_dir = os.path.join(RESULTS_DIR, "figures", sess.id)
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. prediction error δ₁, reward-window firing, gamble trials -----------
    delta1 = load_hgf_trajectory_column(sess.id, len(trials), "delta1")
    if delta1 is None:
        raise SystemExit(f"No HGF trajectory CSV for {sess.id} — run analysis.hgf.run first.")
    delta1 = np.where(gamble_mask, delta1, np.nan)   # PE defined only on observed gamble trials
    rates_d = trial_firing_rates(trains, trials, sr,
                                 window="reward_to_end", trial_mask=gamble_mask)
    print(f"\nTop {args.top} neurons tracking prediction error delta1 (reward_to_end):")
    top_d = rank_by_abs_r(delta1, rates_d, labels, args.top)
    panel_d = grid(sess, "fr_vs_delta1", neurons=top_d, ncols=3,
                   window="reward_to_end", n_bins=8, by_condition=True)
    panel_d.save(os.path.join(out_dir, f"fr_vs_delta1_top{args.top}_reward_to_end.png"))

    # --- 2. perceived probability p̂, cue→reward firing, responding trials -----
    phat = load_hgf_perceived_prob(sess.id, len(trials))
    rates_p = trial_firing_rates(trains, trials, sr,
                                 window="cue_to_reward", trial_mask=resp)
    print(f"\nTop {args.top} neurons tracking perceived probability p-hat (cue_to_reward):")
    top_p = rank_by_abs_r(phat, rates_p, labels, args.top)
    panel_p = grid(sess, "fr_vs_hgf_p", neurons=top_p, ncols=3,
                   window="cue_to_reward", n_bins=8, by_condition=False)
    panel_p.save(os.path.join(out_dir, f"fr_vs_hgf_p_top{args.top}_cue_to_reward.png"))


if __name__ == "__main__":
    main()
