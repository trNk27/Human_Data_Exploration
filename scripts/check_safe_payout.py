"""Check whether the Safe arm always pays out (justifying a fixed Q_safe).

For each session: distribution of P_SmallReward_Safe, and on safe-arm responding
trials whether Rewarded is always 1. Also the gamble arm for contrast.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from session import Session
from analysis.hgf.data import list_sessions

for sid in list_sessions():
    sess = Session(sid)
    t = sess.trials
    resp = (t["NotResponding"] == 0).to_numpy()
    arm = t["ChosenArm_G1S0"].to_numpy()
    rew = t["Rewarded"].to_numpy()

    p_safe = t["P_SmallReward_Safe"].to_numpy()
    p_gamb = t["P_BigReward_Gamble"].to_numpy()

    safe_resp = resp & (arm == 0)
    gamb_resp = resp & (arm == 1)

    safe_rew_rate = rew[safe_resp].mean() if safe_resp.sum() else float("nan")
    gamb_rew_rate = rew[gamb_resp].mean() if gamb_resp.sum() else float("nan")

    print(f"\n=== {sid} ===")
    print(f"  P_SmallReward_Safe  unique: {np.unique(p_safe[resp])}")
    print(f"  P_BigReward_Gamble  unique: {np.unique(p_gamb[resp])}")
    print(f"  safe-arm trials:   n={safe_resp.sum():4d}  rewarded rate={safe_rew_rate:.3f}")
    print(f"  gamble-arm trials: n={gamb_resp.sum():4d}  rewarded rate={gamb_rew_rate:.3f}")
