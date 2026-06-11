"""
Independent verification of neural summary statistics.
Reads the 8 per-session neuron_summary CSVs and recomputes all claimed numbers.
"""

import sys
import os
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
REPO = r"D:\_work_mstammler\Human_Data"
RESULTS = os.path.join(REPO, "results")
SESSIONS = [
    "20250521", "20250602", "20250605", "20250703",
    "20250707", "20250709", "20250710", "20250714",
]

# ── load all CSVs ──────────────────────────────────────────────────────────────
frames = []
for s in SESSIONS:
    path = os.path.join(RESULTS, f"neuron_summary_{s}.csv")
    df = pd.read_csv(path)
    df["session"] = s          # make sure column exists (some CSVs already have it)
    frames.append(df)

df_all = pd.concat(frames, ignore_index=True)
print(f"\nLoaded {len(df_all)} rows total")
print(f"Columns: {list(df_all.columns)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: standard Benjamini-Hochberg FDR, returns boolean mask (True = reject)
# ─────────────────────────────────────────────────────────────────────────────
def bh_reject(pvals, q=0.05):
    """Return boolean array: True where BH(q) rejects H0.  NaN p-values never rejected."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.zeros(n, dtype=bool)
    rank = np.empty(n, dtype=int)
    order = np.argsort(pvals)
    rank[order] = np.arange(1, n + 1)
    threshold = (rank / n) * q
    # largest k such that p(k) <= (k/m)*q
    # a hypothesis is rejected iff its p <= threshold for the *largest* rejected rank
    sorted_p = pvals[order]
    sorted_thr = np.arange(1, n + 1) / n * q
    below = sorted_p <= sorted_thr
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_k = np.where(below)[0].max()          # largest index (0-based) that satisfies
    reject_mask = np.zeros(n, dtype=bool)
    reject_mask[order[: max_k + 1]] = True    # all with rank <= max_k+1 are rejected
    return reject_mask


def fdr_per_group(df, p_col, group_cols=("session",), q=0.05):
    """Add a boolean FDR-reject column named p_col + '_fdr' to df (in-place)."""
    col_out = p_col + "_fdr"
    df[col_out] = False
    for keys, grp in df.groupby(list(group_cols)):
        pv = grp[p_col].values.astype(float)
        rej = bh_reject(pv, q=q)
        df.loc[grp.index, col_out] = rej
    return col_out


# =============================================================================
# CLAIM 1 — counts
# =============================================================================
print("=" * 70)
print("CLAIM 1: Total neurons and region / unit_type counts")
print("=" * 70)

total = len(df_all)
region_counts = df_all["region"].value_counts()
unit_counts   = df_all["unit_type"].value_counts()

claimed = {"total": 1845, "MFG": 702, "IFG": 469, "SMG": 561, "AG": 113,
           "su": 1386, "mu": 459}
mine    = {"total": total,
           "MFG": region_counts.get("MFG", 0),
           "IFG": region_counts.get("IFG", 0),
           "SMG": region_counts.get("SMG", 0),
           "AG":  region_counts.get("AG",  0),
           "su":  unit_counts.get("su",  0),
           "mu":  unit_counts.get("mu",  0)}

for k, cv in claimed.items():
    mv = mine[k]
    verdict = "PASS" if cv == mv else "FAIL"
    print(f"  {k:10s}  claimed={cv}  computed={mv}  -> {verdict}")

print(f"\n  Per-session counts:")
for s, grp in df_all.groupby("session"):
    print(f"    {s}: {len(grp)} neurons")

# =============================================================================
# CLAIM 2 — raw p<0.05 responsiveness rates
# =============================================================================
print("\n" + "=" * 70)
print("CLAIM 2: Raw p<0.05 responsiveness rates")
print("=" * 70)

events_1samp = {
    "trial_start":      "p_zeta_trial_start",
    "cue":              "p_zeta_cue",
    "cue_to_reward":    "p_zeta_cue_to_reward",
    "reward":           "p_zeta_reward",
}
claimed_raw = {"trial_start": 89.9, "cue": 92.5, "cue_to_reward": 93.7, "reward": 77.8}

for ev, col in events_1samp.items():
    pct = (df_all[col] < 0.05).mean() * 100
    cv  = claimed_raw[ev]
    verdict = "PASS" if abs(pct - cv) <= 2.0 else "FAIL"
    print(f"  {ev:18s}  claimed={cv:.1f}%  computed={pct:.2f}%  -> {verdict}")

# =============================================================================
# CLAIM 3 — BH-FDR corrected responsiveness (per session per event)
# =============================================================================
print("\n" + "=" * 70)
print("CLAIM 3: BH-FDR (q<0.05) responsiveness rates (per session)")
print("=" * 70)

claimed_fdr = {"trial_start": 89.4, "cue": 92.0, "cue_to_reward": 93.4, "reward": 75.7}

for ev, col in events_1samp.items():
    out_col = fdr_per_group(df_all, col, group_cols=("session",))
    pct = df_all[out_col].mean() * 100
    cv  = claimed_fdr[ev]
    verdict = "PASS" if abs(pct - cv) <= 2.0 else "FAIL"
    print(f"  {ev:18s}  claimed={cv:.1f}%  computed={pct:.2f}%  -> {verdict}")

# --- p-value distribution for reward (sanity check) ---
print("\n  p_zeta_reward distribution:")
p_rw = df_all["p_zeta_reward"].dropna()
for thr in [1e-3, 1e-5, 1e-10, 1e-20]:
    frac = (p_rw < thr).mean() * 100
    print(f"    < {thr:.0e} : {frac:.1f}%")
print(f"    median   : {p_rw.median():.4e}")
print(f"    mean     : {p_rw.mean():.4e}")

# =============================================================================
# CLAIM 4 — outcome selectivity FDR rates
# =============================================================================
print("\n" + "=" * 70)
print("CLAIM 4: Outcome-selectivity FDR q<0.05 rates")
print("=" * 70)

two_samp_cols = {
    "rew_outcome":    "p_zeta2_rew_outcome",
    "choice_outcome": "p_zeta2_choice_outcome",
}
claimed_sel = {"rew_outcome": 52.0, "choice_outcome": 46.3}

for key, col in two_samp_cols.items():
    out_col = fdr_per_group(df_all, col, group_cols=("session",))
    pct = df_all[out_col].mean() * 100
    cv  = claimed_sel[key]
    verdict = "PASS" if abs(pct - cv) <= 2.0 else "FAIL"
    print(f"  {key:22s}  claimed={cv:.1f}%  computed={pct:.2f}%  -> {verdict}")

# =============================================================================
# CLAIM 5 — regional dissociation (FDR q<0.05)
# =============================================================================
print("\n" + "=" * 70)
print("CLAIM 5: Regional dissociation (FDR q<0.05)")
print("=" * 70)

# FDR columns already added above (same col names)
rew_fdr_col    = "p_zeta2_rew_outcome_fdr"
choice_fdr_col = "p_zeta2_choice_outcome_fdr"

claimed_rew_region    = {"MFG": 61.5, "IFG": 60.3, "SMG": 36.4, "AG": 36.3}
claimed_choice_region = {"AG": 62.8, "IFG": 53.7, "MFG": 50.6, "SMG": 31.6}

print("\n  REWARD contrast (G+R vs G+N):")
for region in ["MFG", "IFG", "SMG", "AG"]:
    sub = df_all[df_all["region"] == region]
    pct = sub[rew_fdr_col].mean() * 100
    cv  = claimed_rew_region[region]
    verdict = "PASS" if abs(pct - cv) <= 2.0 else "FAIL"
    print(f"    {region:5s}  n={len(sub):4d}  claimed={cv:.1f}%  computed={pct:.2f}%  -> {verdict}")

# qualitative check: frontal > parietal
mfg_r = df_all[df_all["region"] == "MFG"][rew_fdr_col].mean()
ifg_r = df_all[df_all["region"] == "IFG"][rew_fdr_col].mean()
smg_r = df_all[df_all["region"] == "SMG"][rew_fdr_col].mean()
ag_r  = df_all[df_all["region"] == "AG" ][rew_fdr_col].mean()
frontal_gt_parietal_rew = (min(mfg_r, ifg_r) > max(smg_r, ag_r))
print(f"  Qualitative: frontal > parietal for REWARD? {frontal_gt_parietal_rew} -> {'PASS' if frontal_gt_parietal_rew else 'FAIL'}")

print("\n  CHOICE contrast (G+R vs S+R):")
for region in ["MFG", "IFG", "SMG", "AG"]:
    sub = df_all[df_all["region"] == region]
    pct = sub[choice_fdr_col].mean() * 100
    cv  = claimed_choice_region.get(region, None)
    verdict = "PASS" if cv is not None and abs(pct - cv) <= 2.0 else "FAIL"
    print(f"    {region:5s}  n={len(sub):4d}  claimed={cv:.1f}%  computed={pct:.2f}%  -> {verdict}")

# qualitative: AG highest for choice
choice_pcts = {r: df_all[df_all["region"] == r][choice_fdr_col].mean()
               for r in ["MFG", "IFG", "SMG", "AG"]}
ag_highest = choice_pcts["AG"] == max(choice_pcts.values())
print(f"  Qualitative: AG highest for CHOICE? {ag_highest} -> {'PASS' if ag_highest else 'FAIL'}")
print(f"  Ranking: {sorted(choice_pcts.items(), key=lambda x: -x[1])}")

# =============================================================================
# CLAIM 6 — preference balance & mean SI for reward-selective neurons
# =============================================================================
print("\n" + "=" * 70)
print("CLAIM 6: Preference balance and mean SI (reward-selective, FDR q<0.05)")
print("=" * 70)

sig_rew = df_all[df_all[rew_fdr_col] == True].copy()
pref_counts = sig_rew["pref_rew_outcome"].value_counts()
mean_si     = sig_rew["SI_rew_outcome"].mean()

claimed_nr  = 514
claimed_rew = 445
claimed_si  = 0.0

nr_val  = pref_counts.get("non-rewarded", 0)
rew_val = pref_counts.get("rewarded", 0)

print(f"  pref_rew_outcome counts:")
print(f"    non-rewarded: claimed={claimed_nr}  computed={nr_val}  -> {'PASS' if abs(nr_val - claimed_nr) <= 15 else 'FAIL'}")
print(f"    rewarded:     claimed={claimed_rew}  computed={rew_val}  -> {'PASS' if abs(rew_val - claimed_rew) <= 15 else 'FAIL'}")
print(f"  mean SI_rew_outcome: claimed~{claimed_si:.2f}  computed={mean_si:.4f}  -> {'PASS' if abs(mean_si - claimed_si) <= 0.05 else 'FAIL'}")

# Additional distribution detail
print(f"\n  SI_rew_outcome full distribution (all neurons):")
si_all = df_all["SI_rew_outcome"].dropna()
print(f"    mean={si_all.mean():.4f}  median={si_all.median():.4f}  "
      f"std={si_all.std():.4f}  min={si_all.min():.4f}  max={si_all.max():.4f}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY — all verdicts")
print("=" * 70)

print("""
Claim  Description                              Mine    Yours   Verdict
-----  -----------------------------------------  ------  ------  -------
(see per-claim output above for details)
""")
print("Script complete.")
