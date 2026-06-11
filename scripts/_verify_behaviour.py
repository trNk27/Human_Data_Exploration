"""
Independent behavioural verification script.
Loads raw Trials_Sync.mat for each session and recomputes all claimed metrics.
No trust in existing analysis code — raw scipy.io only.
"""
import os
import numpy as np
import scipy.io

DATA_DIR = r"D:\_work_mstammler\Human_Data"
SESSIONS = ["20250521", "20250602", "20250605", "20250703",
            "20250707", "20250709", "20250710", "20250714"]

# ── Column indices (0-based) ─────────────────────────────────────────────────
COL_BLOCK       = 3
COL_P_BIG       = 5
COL_P_SMALL     = 6
COL_NOT_RESP    = 10
COL_CHOSEN_SIDE = 11   # unreliable per README — not used
COL_CHOSEN_ARM  = 12   # 1=Gamble, 0=Safe
COL_REWARDED    = 13

def load_trials(session_id):
    path = os.path.join(DATA_DIR, session_id, "Trials_Sync.mat")
    mat  = scipy.io.loadmat(path)
    arr  = mat["Trials_Sync"]          # (nTrials, 19)
    return arr

# ── Per-session stats ─────────────────────────────────────────────────────────
gamble_rates    = []
reward_rates    = []
p_levels_all    = []          # set union across sessions
p_gamble_by_p   = {0.1: [], 0.2: [], 0.4: [], 0.8: []}  # pooled lists of choices
session_corrs   = []

# win-stay / lose-shift: accumulate across sessions
ws_num = ls_num = 0
ws_den = ls_den = 0

print(f"{'Session':<12} {'nTrials':>8} {'nResp':>7} {'GambleRate':>12} {'RewardRate':>12}")
print("-" * 60)

for sid in SESSIONS:
    arr = load_trials(sid)
    n_total = arr.shape[0]

    # Responding trials mask
    not_resp   = arr[:, COL_NOT_RESP].astype(int)
    resp_mask  = not_resp == 0
    arr_r      = arr[resp_mask]        # responding trials only

    chosen_arm = arr_r[:, COL_CHOSEN_ARM].astype(int)   # 1=Gamble, 0=Safe
    rewarded   = arr_r[:, COL_REWARDED].astype(int)
    p_big      = arr_r[:, COL_P_BIG]
    block      = arr_r[:, COL_BLOCK]

    n_resp = len(arr_r)

    # ── Claim 1: gamble rate ──────────────────────────────────────────────────
    gr = chosen_arm.mean()
    gamble_rates.append(gr)

    # ── Claim 2: reward rate ─────────────────────────────────────────────────
    rr = rewarded.mean()
    reward_rates.append(rr)

    # ── Claim 5: P levels ────────────────────────────────────────────────────
    levels = np.unique(np.round(p_big, 4))
    p_levels_all.append(set(levels.tolist()))

    # ── Claim 3: P(gamble) by scheduled P(big) ───────────────────────────────
    for p_val in [0.1, 0.2, 0.4, 0.8]:
        mask = np.abs(p_big - p_val) < 1e-6
        if mask.sum() > 0:
            p_gamble_by_p[p_val].extend(chosen_arm[mask].tolist())

    # Within-session correlation between scheduled-P and P(gamble)
    # Use per-(session, p_level) means
    sess_p_means = []
    sess_g_means = []
    for p_val in [0.1, 0.2, 0.4, 0.8]:
        mask = np.abs(p_big - p_val) < 1e-6
        if mask.sum() > 0:
            sess_p_means.append(p_val)
            sess_g_means.append(chosen_arm[mask].mean())
    if len(sess_p_means) >= 2:
        corr = np.corrcoef(sess_p_means, sess_g_means)[0, 1]
        session_corrs.append(corr)

    # ── Claim 4: win-stay / lose-shift ───────────────────────────────────────
    # Iterate over responding trials in sequence
    # After a GAMBLE trial:
    #   win  (rewarded=1) → did they stay (gamble again) next responding trial?
    #   loss (rewarded=0) → did they shift (go safe) next responding trial?
    for i in range(len(arr_r) - 1):
        if chosen_arm[i] == 1:               # gamble trial
            next_arm = chosen_arm[i + 1]
            if rewarded[i] == 1:             # win
                ws_num += int(next_arm == 1)
                ws_den += 1
            else:                            # loss
                ls_num += int(next_arm == 0)
                ls_den += 1

    print(f"{sid:<12} {n_total:>8} {n_resp:>7} {gr:>12.4f} {rr:>12.4f}")

# ── Aggregate results ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
overall_gr = np.mean(gamble_rates)
overall_rr = np.mean(reward_rates)
ws_rate    = ws_num / ws_den if ws_den > 0 else float("nan")
ls_rate    = ls_num / ls_den if ls_den > 0 else float("nan")

print(f"\nCLAIM 1 — Overall gamble rate (mean across sessions): {overall_gr:.4f}")
print(f"  Session range: [{min(gamble_rates):.4f}, {max(gamble_rates):.4f}]")
print(f"  Per session: {[round(g,4) for g in gamble_rates]}")

print(f"\nCLAIM 2 — Overall reward rate (mean across sessions): {overall_rr:.4f}")
print(f"  Per session: {[round(r,4) for r in reward_rates]}")

print(f"\nCLAIM 3 — P(gamble) by scheduled P(big reward):")
p_gamble_means = {}
for p_val in [0.1, 0.2, 0.4, 0.8]:
    choices = p_gamble_by_p[p_val]
    if choices:
        mean_g = np.mean(choices)
        p_gamble_means[p_val] = mean_g
        print(f"  P={p_val:.1f}: n={len(choices):>5}, P(gamble)={mean_g:.4f}")
    else:
        print(f"  P={p_val:.1f}: NO DATA")

# Check monotonicity
vals = [p_gamble_means[p] for p in [0.1, 0.2, 0.4, 0.8]]
monotone = all(vals[i] < vals[i+1] for i in range(len(vals)-1))
print(f"  Monotone increasing? {monotone}")
print(f"  Mean within-session Pearson r(P_sched, P_gamble): {np.mean(session_corrs):.4f}")
print(f"  Per-session correlations: {[round(c,4) for c in session_corrs]}")

print(f"\nCLAIM 4 — Win-stay / Lose-shift (pooled, responding-trial adjacent pairs):")
print(f"  Win-stay  : {ws_num}/{ws_den} = {ws_rate:.4f}")
print(f"  Lose-shift: {ls_num}/{ls_den} = {ls_rate:.4f}")
print(f"  WS >> LS? {ws_rate > ls_rate}")

print(f"\nCLAIM 5 — Unique P(big reward) levels per session:")
union_all = set()
for sid, levels in zip(SESSIONS, p_levels_all):
    print(f"  {sid}: {sorted(levels)}")
    union_all |= levels
print(f"  Union across all sessions: {sorted(union_all)}")

# ── Extra: check safe-arm P and anomalies ────────────────────────────────────
print("\n--- ADDITIONAL CHECKS ---")
for sid in SESSIONS:
    arr = load_trials(sid)
    not_resp  = arr[:, COL_NOT_RESP].astype(int)
    resp_mask = not_resp == 0
    arr_r     = arr[resp_mask]
    p_small   = arr_r[:, COL_P_SMALL]
    block     = arr_r[:, COL_BLOCK]
    rewarded  = arr_r[:, COL_REWARDED].astype(int)
    chosen    = arr_r[:, COL_CHOSEN_ARM].astype(int)

    # Unique safe-arm probabilities (col 6, P_small = safe P)
    unique_psmall = np.unique(np.round(p_small, 4))
    # Safe-arm unrewarded
    safe_mask        = chosen == 0
    safe_unrewarded  = (rewarded[safe_mask] == 0).sum()
    safe_n           = safe_mask.sum()
    # Anomalous block values
    unique_blocks    = np.unique(block)
    neg_blocks       = unique_blocks[unique_blocks < 0]

    # Non-responding trial count
    n_nonresp = (arr[:, COL_NOT_RESP] == 1).sum()

    print(f"\n{sid}:")
    print(f"  Non-responding trials: {n_nonresp}/{arr.shape[0]}")
    print(f"  P_small (col6) unique values: {unique_psmall}")
    print(f"  Safe-arm unrewarded: {safe_unrewarded}/{safe_n} ({safe_unrewarded/safe_n:.3f} if safe_n>0 else '-')")
    print(f"  Block values incl negatives: {sorted(unique_blocks.tolist())}")

print("\nDone.")
