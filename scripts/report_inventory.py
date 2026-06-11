"""Cross-session data inventory for the 10-page report.

Loads every YYYYMMDD session and prints a structured summary:
  - sampling rate, n trials, n neurons, areas + unit types
  - behavioural summary: responding rate, gamble rate, reward rate,
    blocks, gamble-arm probabilities present, reward amounts
  - per-condition trial counts (G+R / G+N / S+R and the ignored S+N)

Run:  python scripts/report_inventory.py
Writes a tidy CSV to results/report/session_inventory.csv
"""
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from utils import REPO_ROOT, load_sr, load_stmtx, load_trials_sync, CONDITIONS

SESSION_RE = re.compile(r"^\d{8}$")
OUT_DIR = os.path.join(REPO_ROOT, "results", "report")


def sessions():
    return sorted(d for d in os.listdir(REPO_ROOT)
                  if SESSION_RE.match(d) and os.path.isdir(os.path.join(REPO_ROOT, d)))


def area_of(label):
    # label format: "unit | AREA electrode (su/mu)"
    try:
        right = label.split("|", 1)[1].strip()
        return right.split()[0]
    except Exception:
        return "?"


def unit_type_of(label):
    m = re.search(r"\((su|mu)\)", label)
    return m.group(1) if m else "?"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for sid in sessions():
        ddir = os.path.join(REPO_ROOT, sid)
        sr = int(load_sr(data_dir=ddir)["SamplingRate_Hz"].iloc[0])
        stm = load_stmtx(data_dir=ddir)
        tr = load_trials_sync(data_dir=ddir)

        labels = list(stm.columns)
        areas = [area_of(l) for l in labels]
        utypes = [unit_type_of(l) for l in labels]
        area_counts = pd.Series(areas).value_counts().to_dict()
        su = sum(1 for u in utypes if u == "su")
        mu = sum(1 for u in utypes if u == "mu")

        # spikes per neuron
        spikes_per = [stm[c].dropna().shape[0] for c in stm.columns]
        rec_dur = float(np.nanmax(stm.to_numpy())) if stm.size else np.nan

        n_trials = len(tr)
        responding = (tr["NotResponding"] == 0)
        n_resp = int(responding.sum())
        arm = tr["ChosenArm_G1S0"].to_numpy()
        rew = tr["Rewarded"].to_numpy()
        gamble_rate = float(np.mean(arm[responding] == 1)) if n_resp else np.nan
        reward_rate = float(np.mean(rew[responding] == 1)) if n_resp else np.nan

        # condition counts
        cond_counts = {}
        for name, cfg in CONDITIONS.items():
            cond_counts[name] = int(((responding) & (arm == cfg["arm"]) & (rew == cfg["rewarded"])).sum())
        # ignored safe+no
        cond_counts["S+N"] = int(((responding) & (arm == 0) & (rew == 0)).sum())

        # blocks (clean out matlab artefacts like -990)
        blocks_raw = tr["Block"].to_numpy()
        blocks_clean = blocks_raw[(blocks_raw >= 0) & (blocks_raw < 1000)]
        n_blocks = len(np.unique(blocks_clean))

        # gamble-arm probabilities used (schedule)
        pbig = tr["P_BigReward_Gamble"].to_numpy()
        pbig_vals = sorted(set(np.round(pbig[np.isfinite(pbig)], 3).tolist()))
        psafe = tr["P_SmallReward_Safe"].to_numpy()
        psafe_vals = sorted(set(np.round(psafe[np.isfinite(psafe)], 3).tolist()))

        # reward amounts  (these columns are all-NaN in the exports — summarise compactly)
        def amt_summary(col):
            v = tr[col].to_numpy()
            finite = v[np.isfinite(v)]
            if finite.size == 0:
                return "all NaN"
            return str(sorted(set(np.round(finite, 3).tolist())))
        amt_big = amt_summary("Amount_BigReward_Gamble")
        amt_small = amt_summary("Amount_SmallReward_Safe")

        rows.append(dict(
            session=sid, sr=sr, n_trials=n_trials, n_responding=n_resp,
            n_neurons=len(labels), n_su=su, n_mu=mu,
            rec_dur_min=round(rec_dur / 60, 1) if np.isfinite(rec_dur) else np.nan,
            median_spikes=int(np.median(spikes_per)) if spikes_per else 0,
            gamble_rate=round(gamble_rate, 3), reward_rate=round(reward_rate, 3),
            n_blocks=n_blocks, **{f"n_{k}": v for k, v in cond_counts.items()},
            areas=str(area_counts),
            pbig_gamble=str(pbig_vals), psafe=str(psafe_vals),
            amt_big=amt_big, amt_small=amt_small,
        ))

        print(f"\n=== {sid} ===")
        print(f"  SR={sr} Hz | rec ~{rows[-1]['rec_dur_min']} min | trials={n_trials} (responding {n_resp})")
        print(f"  neurons={len(labels)} (su={su}, mu={mu}) | median spikes/neuron={rows[-1]['median_spikes']}")
        print(f"  areas: {area_counts}")
        print(f"  gamble_rate={gamble_rate:.2f} reward_rate={reward_rate:.2f} blocks={n_blocks}")
        print(f"  conditions: " + ", ".join(f"{k}={v}" for k, v in cond_counts.items()))
        print(f"  P(big|gamble) values: {pbig_vals}")
        print(f"  P(safe) values: {psafe_vals}")
        print(f"  amounts big={amt_big} small={amt_small}")

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "session_inventory.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    # totals
    print("\n=== TOTALS ===")
    print(f"  sessions={len(df)} | total neurons={df['n_neurons'].sum()} "
          f"(su={df['n_su'].sum()}, mu={df['n_mu'].sum()}) | total trials={df['n_trials'].sum()}")
    # area pooled
    from collections import Counter
    pooled = Counter()
    for sid in sessions():
        stm = load_stmtx(data_dir=os.path.join(REPO_ROOT, sid))
        for l in stm.columns:
            pooled[area_of(l)] += 1
    print(f"  pooled areas: {dict(pooled.most_common())}")


if __name__ == "__main__":
    main()
