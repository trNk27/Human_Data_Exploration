"""Inspect the behavioural schedule across all sessions for HGF modelling.

Reports, per session: trial counts (total / responding / gamble / safe),
the gamble-arm reward-probability schedule (unique values, #changes, block
structure), the safe-arm probability, the reward amounts actually stored in
the data, choice fractions, and the behavioural-clock span. The cross-session
summary is what informs the belief-continuity switch (reset vs carry-over) and
confirms whether the hardcoded reward magnitudes (safe=1, gamble=4) match.

Run: python scripts/inspect_schedule.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import REPO_ROOT, load_trials_sync, load_sr


def session_ids() -> list[str]:
    ids = [d for d in os.listdir(REPO_ROOT)
           if len(d) == 8 and d.isdigit() and
           os.path.isdir(os.path.join(REPO_ROOT, d))]
    return sorted(ids)


def runs(values: np.ndarray) -> int:
    """Number of value changes (transitions) in a 1-D sequence."""
    if len(values) < 2:
        return 0
    return int(np.sum(values[1:] != values[:-1]))


def inspect(sid: str) -> dict:
    data_dir = os.path.join(REPO_ROOT, sid)
    trials = load_trials_sync(data_dir=data_dir)
    sr = int(load_sr(data_dir=data_dir)["SamplingRate_Hz"].iloc[0])

    n = len(trials)
    responding = (trials["NotResponding"] == 0).to_numpy()
    arm = trials["ChosenArm_G1S0"].to_numpy()
    rew = trials["Rewarded"].to_numpy()
    p_gamble = trials["P_BigReward_Gamble"].to_numpy()
    p_safe = trials["P_SmallReward_Safe"].to_numpy()
    amt_gamble = trials["Amount_BigReward_Gamble"].to_numpy()
    amt_safe = trials["Amount_SmallReward_Safe"].to_numpy()
    block = trials["Block"].to_numpy()
    t_start = trials["TrialStart_s"].to_numpy()
    t_end = trials["TrialEnd_s"].to_numpy()

    resp = responding
    n_resp = int(resp.sum())
    n_gamble = int(np.sum(resp & (arm == 1)))
    n_safe = int(np.sum(resp & (arm == 0)))

    # Schedule of the gamble probability over responding trials
    pg_resp = p_gamble[resp]
    uniq_pg = np.unique(pg_resp[np.isfinite(pg_resp)])

    # Clean block labels (note: data can have artefact blocks like -990)
    block_resp = block[resp]
    uniq_blocks = np.unique(block_resp[np.isfinite(block_resp)])

    return {
        "session": sid,
        "sr": sr,
        "n_trials": n,
        "n_responding": n_resp,
        "n_gamble": n_gamble,
        "n_safe": n_safe,
        "gamble_frac": n_gamble / n_resp if n_resp else np.nan,
        "reward_rate_gamble": float(np.nanmean(rew[resp & (arm == 1)])) if n_gamble else np.nan,
        "reward_rate_safe": float(np.nanmean(rew[resp & (arm == 0)])) if n_safe else np.nan,
        "pg_unique": uniq_pg,
        "pg_changes": runs(pg_resp),
        "pg_first": float(pg_resp[0]) if n_resp else np.nan,
        "pg_last": float(pg_resp[-1]) if n_resp else np.nan,
        "psafe_unique": np.unique(p_safe[resp][np.isfinite(p_safe[resp])]),
        "amt_gamble_unique": np.unique(amt_gamble[np.isfinite(amt_gamble)]),
        "amt_safe_unique": np.unique(amt_safe[np.isfinite(amt_safe)]),
        "n_blocks": len(uniq_blocks),
        "blocks": uniq_blocks,
        "t_start_first": float(np.nanmin(t_start)),
        "t_end_last": float(np.nanmax(t_end)),
        "span_min": (float(np.nanmax(t_end)) - float(np.nanmin(t_start))) / 60.0,
    }


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    rows = []
    for sid in session_ids():
        info = inspect(sid)
        rows.append(info)
        print("=" * 78)
        print(f"SESSION {sid}   (SR={info['sr']} Hz)")
        print(f"  trials: {info['n_trials']} total | {info['n_responding']} responding "
              f"| {info['n_gamble']} gamble | {info['n_safe']} safe "
              f"(gamble frac {info['gamble_frac']:.2f})")
        print(f"  reward rate: gamble={info['reward_rate_gamble']:.3f}  "
              f"safe={info['reward_rate_safe']:.3f}")
        print(f"  P(gamble big reward): {info['pg_unique']}")
        print(f"     changes over trials: {info['pg_changes']}  "
              f"(first={info['pg_first']}, last={info['pg_last']})")
        print(f"  P(safe small reward): {info['psafe_unique']}")
        print(f"  reward amounts: gamble={info['amt_gamble_unique']}  "
              f"safe={info['amt_safe_unique']}")
        print(f"  blocks: {info['n_blocks']}  -> {info['blocks']}")
        print(f"  behavioural-clock span: {info['span_min']:.1f} min")

    print("\n" + "#" * 78)
    print("CROSS-SESSION SUMMARY")
    print("#" * 78)
    summary = pd.DataFrame([{
        "session": r["session"],
        "n_resp": r["n_responding"],
        "n_gamble": r["n_gamble"],
        "n_safe": r["n_safe"],
        "gamble_frac": round(r["gamble_frac"], 2),
        "pg_changes": r["pg_changes"],
        "pg_first": r["pg_first"],
        "pg_last": r["pg_last"],
        "n_pg_levels": len(r["pg_unique"]),
        "amt_g": list(r["amt_gamble_unique"]),
        "amt_s": list(r["amt_safe_unique"]),
    } for r in rows])
    print(summary.to_string(index=False))

    # Aggregate schedule check: do all sessions share the same pg level set?
    all_levels = [tuple(np.round(r["pg_unique"], 3)) for r in rows]
    print("\nGamble-prob level sets per session:")
    for r, lv in zip(rows, all_levels):
        print(f"  {r['session']}: {lv}")
    print("\nMin/median/max responding trials:",
          int(np.min([r["n_responding"] for r in rows])),
          int(np.median([r["n_responding"] for r in rows])),
          int(np.max([r["n_responding"] for r in rows])))


if __name__ == "__main__":
    main()
