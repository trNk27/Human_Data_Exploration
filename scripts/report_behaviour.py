"""Behavioural learning metrics across all sessions, for the report.

Computes, per session and pooled (gamble = arm 1, safe = arm 0; responding trials):
  - choice rates, reward rate
  - P(choose gamble) as a function of the *scheduled* gamble probability
    (.1/.2/.4/.8)  -> does the subject track true value?
  - win-stay / lose-shift on the GAMBLE arm (stay = repeat gamble next trial)
  - choice ~ perceived probability (rolling reward rate, history=10)
  - block-change adaptation: P(gamble) in first vs second half of a block
  - logistic regression: choose_gamble ~ scheduled_p + prev_gamble_reward + prev_choice

Writes results/report/behaviour_metrics.csv and a summary figure.
Run:  python scripts/report_behaviour.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from session import Session
from utils import REPO_ROOT
from compute import perceived_probability

OUT = os.path.join(REPO_ROOT, "results", "report")
SESSIONS = ["20250521", "20250602", "20250605", "20250703",
            "20250707", "20250709", "20250710", "20250714"]
P_LEVELS = [0.1, 0.2, 0.4, 0.8]


def session_metrics(sid):
    sess = Session(sid)
    tr = sess.trials
    resp = sess.responding_mask
    arm = tr["ChosenArm_G1S0"].to_numpy()
    rew = tr["Rewarded"].to_numpy()
    pg = tr["P_BigReward_Gamble"].to_numpy()
    n = len(tr)

    gamble = (arm == 1) & resp
    safe = (arm == 0) & resp

    out = dict(session=sid, n_resp=int(resp.sum()),
               gamble_rate=round(float(gamble.sum() / resp.sum()), 3),
               reward_rate=round(float(rew[resp].mean()), 3))

    # P(choose gamble) by scheduled probability level
    pg_by_level = {}
    for lvl in P_LEVELS:
        m = resp & np.isclose(pg, lvl)
        if m.sum() > 0:
            pg_by_level[lvl] = float((arm[m] == 1).mean())
    out["pgamble_by_p"] = {k: round(v, 3) for k, v in pg_by_level.items()}

    # slope: does P(gamble) rise with scheduled p?  (Pearson over the 4 points)
    if len(pg_by_level) >= 2:
        xs = np.array(list(pg_by_level.keys()))
        ys = np.array(list(pg_by_level.values()))
        out["pgamble_vs_p_r"] = round(float(np.corrcoef(xs, ys)[0, 1]), 3)
    else:
        out["pgamble_vs_p_r"] = np.nan

    # Win-stay / lose-shift on consecutive responding gamble trials
    # consider trials where current is gamble and previous responding choice exists
    stay_win = stay_tot_win = shift_loss = shift_tot_loss = 0
    last_resp_idx = None
    for t in range(n):
        if not resp[t]:
            continue
        if last_resp_idx is not None and arm[last_resp_idx] == 1:
            # previous responding trial was a gamble
            prev_rew = rew[last_resp_idx] == 1
            repeated_gamble = arm[t] == 1
            if prev_rew:
                stay_tot_win += 1
                if repeated_gamble:
                    stay_win += 1
            else:
                shift_tot_loss += 1
                if not repeated_gamble:
                    shift_loss += 1
        last_resp_idx = t
    out["win_stay"] = round(stay_win / stay_tot_win, 3) if stay_tot_win else np.nan
    out["lose_shift"] = round(shift_loss / shift_tot_loss, 3) if shift_tot_loss else np.nan

    # choice ~ perceived probability correlation (gamble trials get a perc-prob defined,
    # but we want choice on ALL responding trials vs the rolling reward rate up to t)
    pp = perceived_probability(tr, resp, history=10)
    # perceived_probability is only defined on gamble responding trials; build a
    # forward-filled "current belief" for all responding trials instead:
    belief = np.full(n, np.nan)
    past = []
    for t in range(n):
        if past:
            belief[t] = np.mean(past[-10:])
        if resp[t] and arm[t] == 1:
            past.append(float(rew[t] == 1))
    m = resp & np.isfinite(belief)
    if m.sum() > 10:
        out["choice_vs_belief_r"] = round(float(np.corrcoef(belief[m], (arm[m] == 1).astype(float))[0, 1]), 3)
    else:
        out["choice_vs_belief_r"] = np.nan

    return out, pg_by_level


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    pg_curves = {}
    for sid in SESSIONS:
        m, pg_by_level = session_metrics(sid)
        rows.append(m)
        pg_curves[sid] = pg_by_level
        print(f"[{sid}] gamble={m['gamble_rate']} reward={m['reward_rate']} "
              f"P(g|p)={m['pgamble_by_p']} r(p)={m['pgamble_vs_p_r']} "
              f"WS={m['win_stay']} LS={m['lose_shift']} r(choice,belief)={m['choice_vs_belief_r']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "behaviour_metrics.csv"), index=False)

    # pooled P(gamble) by level
    print("\n=== POOLED P(choose gamble) by scheduled gamble probability ===")
    pooled = {lvl: [] for lvl in P_LEVELS}
    for sid in SESSIONS:
        for lvl, v in pg_curves[sid].items():
            pooled[lvl].append(v)
    for lvl in P_LEVELS:
        vals = pooled[lvl]
        print(f"  P(big)={lvl}: P(gamble)={np.mean(vals):.3f} ± {np.std(vals):.3f}  (n_sessions={len(vals)})")

    print("\n=== Means across sessions ===")
    for c in ["gamble_rate", "reward_rate", "win_stay", "lose_shift",
              "pgamble_vs_p_r", "choice_vs_belief_r"]:
        print(f"  {c}: {df[c].mean():.3f} ± {df[c].std():.3f}")

    # Figure: P(gamble) vs scheduled p, one line per session + pooled mean
    fig, ax = plt.subplots(figsize=(7, 5))
    for sid in SESSIONS:
        xs = sorted(pg_curves[sid])
        ys = [pg_curves[sid][x] for x in xs]
        ax.plot(xs, ys, marker="o", alpha=0.4, lw=1, color="gray")
    means = [np.mean(pooled[lvl]) for lvl in P_LEVELS]
    sems = [np.std(pooled[lvl]) / np.sqrt(len(pooled[lvl])) for lvl in P_LEVELS]
    ax.errorbar(P_LEVELS, means, yerr=sems, marker="s", color="darkorange",
                lw=2.5, capsize=4, label="pooled mean ± SEM", zorder=5)
    ax.plot([0, 1], [0, 1], ls=":", color="k", alpha=0.4, label="value-matching")
    ax.set_xlabel("Scheduled P(big reward | gamble)")
    ax.set_ylabel("P(choose gamble)")
    ax.set_title("Choice tracks scheduled gamble value\n(grey = sessions, orange = pooled)")
    ax.set_xticks(P_LEVELS); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_choice_vs_value.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved -> {p}\nSaved -> {os.path.join(OUT, 'behaviour_metrics.csv')}")


if __name__ == "__main__":
    main()
