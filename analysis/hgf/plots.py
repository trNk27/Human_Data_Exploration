"""Plotting layer for the HGF analysis package.

All functions return a (fig, axes) tuple and do NOT call plt.show(), so the
caller can save or further modify. Pass ``save_dir`` to auto-save to PNG.

Per-session plots (one call per session):
    plot_trajectory      — perceived-p trajectory with choices + true schedule
    plot_learning_rate   — learning-rate and uncertainty traces
    plot_volatility      — HGF level-3 (log-volatility) trace

Cross-session / summary plots:
    plot_parameter_drift — separate-MAP estimates over sessions
    plot_model_comparison — bar chart of delta-BIC vs RW / RW+stick
    plot_recovery        — scatter true vs recovered per parameter
    plot_power_curve     — power vs shift magnitude per parameter
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, save_dir: Optional[str], name: str) -> None:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, name), dpi=150, bbox_inches="tight")


def _gamble_color(y: np.ndarray) -> list[str]:
    return ["#2196F3" if c == 1 else "#FF9800" for c in y]


# ---------------------------------------------------------------------------
# Per-session plots
# ---------------------------------------------------------------------------

def plot_trajectory(traj: pd.DataFrame,
                    save_dir: Optional[str] = None) -> tuple:
    """Perceived gamble probability over trials with choices overlaid.

    Top panel: p̂ (blue line) vs true schedule (grey step). Choice markers
    (gamble=▲ blue, safe=▼ orange) on the x-axis strip.
    Bottom panel: choice probability from the response model.
    """
    sid = traj["session_id"].iloc[0]
    trial = traj["trial"].to_numpy()
    p_hat = traj["perceived_gamble_prob"].to_numpy()
    p_sched = traj["true_p_schedule"].to_numpy()
    choice = traj["actual_choice"].to_numpy()
    p_g = traj["p_choose_gamble"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5]})

    ax = axes[0]
    ax.step(trial, p_sched, where="post", color="#BBBBBB", lw=1.5,
            label="true P(big reward)", zorder=1)
    ax.plot(trial, p_hat, color="#1565C0", lw=1.8, label="perceived p̂", zorder=2)

    # Choice markers — ticks at y=0 and y=1 margins
    gamble_t = trial[choice == 1]
    safe_t   = trial[choice == 0]
    ax.scatter(gamble_t, np.full_like(gamble_t, -0.04, dtype=float),
               marker="^", s=18, color="#2196F3", clip_on=False, label="gamble chosen")
    ax.scatter(safe_t, np.full_like(safe_t, -0.08, dtype=float),
               marker="v", s=18, color="#FF9800", clip_on=False, label="safe chosen")

    ax.set_ylim(-0.12, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(f"Session {sid} — HGF perceived gamble probability")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.axhline(0.5, color="grey", lw=0.6, ls="--", alpha=0.4)

    ax2 = axes[1]
    ax2.plot(trial, p_g, color="#7B1FA2", lw=1.5, label="P(choose gamble)")
    ax2.axhline(0.5, color="grey", lw=0.6, ls="--", alpha=0.4)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("P(gamble)")
    ax2.set_xlabel("Trial")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    _save(fig, save_dir, f"trajectory_{sid}.png")
    return fig, axes


def plot_learning_rate(traj: pd.DataFrame,
                       save_dir: Optional[str] = None) -> tuple:
    """Learning rate (1/π₂) and belief uncertainty (sa2 = 1/π̂₂) per trial."""
    sid = traj["session_id"].iloc[0]
    trial = traj["trial"].to_numpy()
    lr  = traj["learning_rate"].to_numpy()
    sa2 = traj["sa2"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    axes[0].plot(trial, lr, color="#E53935", lw=1.5)
    axes[0].set_ylabel("Learning rate\n(1/π₂, posterior)")
    axes[0].set_title(f"Session {sid} — HGF learning-rate & uncertainty")

    axes[1].plot(trial, sa2, color="#F57C00", lw=1.5)
    axes[1].set_ylabel("Belief uncertainty\n(sa2 = 1/π̂₂, prior)")
    axes[1].set_xlabel("Trial")

    fig.tight_layout()
    _save(fig, save_dir, f"learning_rate_{sid}.png")
    return fig, axes


def plot_volatility(traj: pd.DataFrame,
                    save_dir: Optional[str] = None) -> tuple:
    """HGF level-3 posterior mean (log-volatility) and level-2 belief mean."""
    sid = traj["session_id"].iloc[0]
    trial = traj["trial"].to_numpy()
    mu3 = traj["mu3"].to_numpy()
    mu2 = traj["mu2"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    axes[0].plot(trial, mu3, color="#00796B", lw=1.5)
    axes[0].set_ylabel("μ₃ (log-volatility)")
    axes[0].set_title(f"Session {sid} — HGF volatility & belief tendency")

    axes[1].plot(trial, mu2, color="#5C6BC0", lw=1.5)
    axes[1].axhline(0, color="grey", lw=0.6, ls="--", alpha=0.4)
    axes[1].set_ylabel("μ₂ (logit p̂)")
    axes[1].set_xlabel("Trial")

    fig.tight_layout()
    _save(fig, save_dir, f"volatility_{sid}.png")
    return fig, axes


# ---------------------------------------------------------------------------
# Cross-session plots
# ---------------------------------------------------------------------------

def plot_parameter_drift(session_ids: list[str],
                         per_session_fits: list[dict],
                         shared_fit: Optional[dict] = None,
                         save_dir: Optional[str] = None,
                         params: Optional[list[str]] = None) -> tuple:
    """Per-session MAP estimates plotted over session index.

    Each panel shows one free parameter; a horizontal dashed line shows the
    shared (complete-pooling) estimate if provided.

    Parameters
    ----------
    per_session_fits : list of dicts (full natural params), one per session.
    shared_fit : optional dict — the shared (pooled) MAP estimate.
    params : which parameters to draw (default omega2, beta, bias). Pass the
        fitted free set to include omega3/kappa for the extended model.
    """
    if params is None:
        params = ["omega2", "beta", "bias"]
    _drift_lab = {"omega2": "ω₂ (tonic volatility)", "omega3": "ω₃ (meta-volatility)",
                  "kappa": "κ (volatility coupling)", "beta": "β (inverse temperature)",
                  "bias": "bias (gamble preference)"}
    labels = [_drift_lab.get(p, p) for p in params]
    colors = [_PARAM_PALETTE.get(p, "#333333") for p in params]

    x = np.arange(len(session_ids))
    fig, axes = plt.subplots(len(params), 1, figsize=(9, 3 * len(params)), sharex=True)
    if len(params) == 1:
        axes = [axes]

    for ax, p, lab, col in zip(axes, params, labels, colors):
        vals = [f.get(p, np.nan) for f in per_session_fits]
        ax.plot(x, vals, "o-", color=col, ms=7, lw=1.5, label="per-session MAP")
        if shared_fit is not None and p in shared_fit:
            ax.axhline(shared_fit[p], color=col, lw=1.2, ls="--", alpha=0.6,
                       label="shared (pooled)")
        ax.set_ylabel(lab)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Session index")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(session_ids, rotation=30, ha="right", fontsize=8)
    axes[0].set_title("Cross-session parameter drift — per-session vs shared MAP")

    fig.tight_layout()
    _save(fig, save_dir, "parameter_drift.png")
    return fig, axes


def plot_model_comparison(comp_df: pd.DataFrame,
                          save_dir: Optional[str] = None) -> tuple:
    """Horizontal bar chart of ΔlogLik/trial and ΔBIC vs HGF.

    comp_df is the output of comparison.compare_models().
    """
    df = comp_df.sort_values("bic").copy()
    models = df["model"].tolist()
    delta_bic = df["delta_bic"].tolist()
    ll_per_trial = df["loglik_per_trial"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, 0.7 * len(models))))

    y = np.arange(len(models))
    # ΔBIC (lower is better — HGF reference is 0)
    bars = axes[0].barh(y, delta_bic, color=["#43A047" if d == 0 else "#E53935" for d in delta_bic])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(models)
    axes[0].axvline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("ΔBIC (lower = better)")
    axes[0].set_title("Model comparison — ΔBIC")

    # loglik per trial
    axes[1].barh(y, ll_per_trial, color="#5C6BC0")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(models)
    axes[1].set_xlabel("log-likelihood per trial")
    axes[1].set_title("Model comparison — fit quality")

    fig.tight_layout()
    _save(fig, save_dir, "model_comparison.png")
    return fig, axes


#: Display labels / palette for every freeable parameter.
_PARAM_LABELS = {"omega2": "ω₂", "omega3": "ω₃", "kappa": "κ", "beta": "β", "bias": "bias"}
_PARAM_PALETTE = {"omega2": "#1565C0", "omega3": "#6A1B9A", "kappa": "#00838F",
                  "beta": "#AD1457", "bias": "#2E7D32"}


def plot_recovery(rec_df: pd.DataFrame,
                  save_dir: Optional[str] = None) -> tuple:
    """Scatter true vs recovered for each free parameter (one panel per param).

    Parameters are inferred from the ``true_*``/``rec_*`` columns present, so the
    figure adapts to the baseline (3) or extended (5) parameter set.
    """
    params = [c[len("true_"):] for c in rec_df.columns if c.startswith("true_")]
    labels = [_PARAM_LABELS.get(p, p) for p in params]
    colors = [_PARAM_PALETTE.get(p, "#333333") for p in params]

    fig, axes = plt.subplots(1, len(params), figsize=(4.3 * len(params), 4))
    if len(params) == 1:
        axes = [axes]

    for ax, p, lab, col in zip(axes, params, labels, colors):
        t = rec_df[f"true_{p}"].to_numpy()
        r = rec_df[f"rec_{p}"].to_numpy()
        ok = np.isfinite(t) & np.isfinite(r)
        t, r = t[ok], r[ok]

        lo, hi = min(t.min(), r.min()), max(t.max(), r.max())
        margin = (hi - lo) * 0.05
        lo -= margin; hi += margin

        ax.scatter(t, r, color=col, alpha=0.6, s=40, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"True {lab}")
        ax.set_ylabel(f"Recovered {lab}")
        corr = float(np.corrcoef(t, r)[0, 1]) if len(t) > 2 else np.nan
        ax.set_title(f"{lab}  r = {corr:.2f}")
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Parameter recovery — HGF (simulate→refit)", y=1.02)
    fig.tight_layout()
    _save(fig, save_dir, "parameter_recovery.png")
    return fig, axes


def plot_power_curve(power_df: pd.DataFrame,
                     save_dir: Optional[str] = None) -> tuple:
    """Power vs shift magnitude, one panel per free parameter."""
    params = power_df["shift_param"].unique().tolist()
    colors = {"omega2": "#1565C0", "beta": "#AD1457"}

    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4),
                             sharey=True)
    if len(params) == 1:
        axes = [axes]

    for ax, p in zip(axes, params):
        sub = power_df[power_df["shift_param"] == p].sort_values("shift_size")
        col = colors.get(p, "#333333")
        ax.plot(sub["shift_size"], sub["power"], "o-", color=col, ms=7, lw=1.8)
        ax.axhline(0.8, color="grey", lw=0.8, ls="--", alpha=0.7, label="80% power")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"Shift size (Δ{p})")
        ax.set_ylabel("Power (prop. correct sign)" if ax is axes[0] else "")
        ax.set_title(f"Power: detecting Δ{p}")
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(alpha=0.3)

    fig.suptitle("Power check — session-to-session parameter shift detection")
    fig.tight_layout()
    _save(fig, save_dir, "power_curve.png")
    return fig, axes


def plot_posterior_overview(posterior_df: "pd.DataFrame",
                            session_ids: list[str],
                            save_dir: Optional[str] = None) -> tuple:
    """Violin plot of posterior marginals per session from the hierarchical model."""
    params = ["omega2", "beta", "bias"]
    labels = ["ω₂", "β", "bias"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(session_ids))

    for ax, p, lab in zip(axes, params, labels):
        data_per_session = [
            posterior_df[posterior_df["session_id"] == sid][p].to_numpy()
            for sid in session_ids
        ]
        # Strip empty sessions
        data_per_session = [d for d in data_per_session if len(d) > 0]
        if not data_per_session:
            continue
        parts = ax.violinplot(data_per_session, positions=x[:len(data_per_session)],
                              showmedians=True, widths=0.6)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax.set_xticks(x[:len(data_per_session)])
        ax.set_xticklabels(session_ids, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(lab)
        ax.set_title(f"Posterior {lab} by session")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Hierarchical model — per-session posterior marginals")
    fig.tight_layout()
    _save(fig, save_dir, "posterior_overview.png")
    return fig, axes
