"""Chi-squared test: brain region x (responsive / non-responsive).

Pools every per-session ZETA responsiveness CSV in
`results/zeta_responsiveness/`, parses the region out of each neuron's
label (`unit<X> | <REGION> ele<Y> (<type>)`) and asks whether the
proportion of significant neurons depends on region.

For each event (cue, response, reward, trial_start) — and for a combined
"responsive to ANY event" view — the script prints:

  * the region x {resp, non-resp} contingency table with row %s,
  * chi2_contingency result (chi2, dof, p),
  * standardized (Pearson) residuals to show *which* regions drive any
    significant effect.

Regions with fewer than `--min-n` neurons in the pooled sample are dropped
before testing (chi-squared expects expected counts >= ~5).

By default it reads the fixed-window ZETA files (`zeta_<event>_<session>.csv`).
Pass `--window-end` to instead analyse the variable-window results
(`zeta_<event>_to_<end>_<session>.csv`) from the per-trial cue->reward test.

Usage:
    python responsive_region.py
    python responsive_region.py --alpha 0.01
    python responsive_region.py --event reward
    python responsive_region.py --min-n 10 --csv
    python responsive_region.py --event cue --window-end reward --plot
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from utils import EVENTS, RESULTS_SUBDIRS

# label format: "unit73 | AG ele059 (su)"  ->  region = "AG"
_REGION_RE = re.compile(r"\|\s*(\S+)\s+ele")


def parse_region(label):
    m = _REGION_RE.search(label)
    return m.group(1) if m else None


def event_stem(event, window_end=None):
    """Filename stem for an event's ZETA CSVs.

    Fixed window:    zeta_<event>            (-> zeta_cue_20250521.csv)
    Variable window: zeta_<event>_to_<end>   (-> zeta_cue_to_reward_20250521.csv)
    """
    return f"zeta_{event}_to_{window_end}" if window_end else f"zeta_{event}"


def load_event(event, window_end=None):
    """Pool all session CSVs for one event. Returns DataFrame with `region` col.

    With `window_end=None` matches the fixed-window files
    `zeta_<event>_<8-digit session>.csv`; with `window_end` set matches the
    variable-window files `zeta_<event>_to_<end>_<session>.csv`. The strict
    8-digit suffix keeps the two variants from contaminating each other.
    """
    stem     = event_stem(event, window_end)
    pattern  = os.path.join(RESULTS_SUBDIRS["responsiveness"], f"{stem}_*.csv")
    strict   = re.compile(rf"{re.escape(stem)}_\d{{8}}\.csv$")
    files    = sorted(f for f in glob.glob(pattern) if strict.search(os.path.basename(f)))
    if not files:
        return None
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["session"] = re.search(r"_(\d{8})\.csv$", os.path.basename(f)).group(1)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["region"] = out["label"].map(parse_region)
    out = out.dropna(subset=["region", "p_zeta"])
    return out


def contingency(df, alpha, min_n):
    """region x {resp, non-resp} table, regions with < min_n dropped."""
    df = df.copy()
    df["responsive"] = df["p_zeta"] < alpha
    tab = (df.groupby("region")["responsive"]
             .value_counts().unstack(fill_value=0))
    # ensure both columns exist
    for c in (True, False):
        if c not in tab.columns:
            tab[c] = 0
    tab = tab.rename(columns={True: "responsive", False: "non_responsive"})
    tab = tab[["responsive", "non_responsive"]]
    tab["total"] = tab.sum(axis=1)
    tab = tab[tab["total"] >= min_n].sort_values("total", ascending=False)
    return tab


def run_test(tab, label):
    print(f"\n=== {label} ===")
    if tab.empty or len(tab) < 2:
        print("  Not enough regions after --min-n filter to run a test.")
        return

    pct = 100 * tab["responsive"] / tab["total"]
    show = tab.assign(**{"% resp": pct.round(1)})
    print(show.to_string())

    obs = tab[["responsive", "non_responsive"]].to_numpy()
    chi2, p, dof, exp = chi2_contingency(obs)
    grand_resp = tab["responsive"].sum()
    grand_tot  = tab["total"].sum()
    print(f"\n  pooled: {grand_resp}/{grand_tot} = {100*grand_resp/grand_tot:.1f}% responsive")
    print(f"  chi2(dof={dof}) = {chi2:.3f}, p = {p:.4g}")

    # standardized Pearson residuals: (obs - exp) / sqrt(exp)
    resid = (obs - exp) / np.sqrt(exp)
    res_df = pd.DataFrame(resid, index=tab.index,
                          columns=["resp_resid", "non_resp_resid"]).round(2)
    flagged = res_df[(res_df.abs() > 2).any(axis=1)]
    if not flagged.empty:
        print("  standardized residuals (|z| > 2):")
        for region, row in flagged.iterrows():
            mark = " *" if abs(row["resp_resid"]) > 2 else ""
            print(f"    {region:>6s}  resp={row['resp_resid']:+.2f}{mark}  "
                  f"non={row['non_resp_resid']:+.2f}")
    else:
        print("  no region has |standardized residual| > 2")


def _chi2_residuals(tab):
    """Return (chi2, p, signed per-cell chi2 contribution).

    The per-cell contribution is `sign(r_resp) * sqrt(r_resp**2 + r_non**2)`,
    where r_X is the standardized Pearson residual on column X. Its square
    is the cell's contribution to the chi2 statistic; the sign tells the
    direction on the *responsive* column (+ = more responsive than expected).
    Squaring and summing these values reproduces the chi2 statistic.
    """
    obs = tab[["responsive", "non_responsive"]].to_numpy()
    result = chi2_contingency(obs)
    chi2, p, exp = result[0], result[1], result[3]
    r_resp = (obs[:, 0] - exp[:, 0]) / np.sqrt(exp[:, 0])
    r_non  = (obs[:, 1] - exp[:, 1]) / np.sqrt(exp[:, 1])
    cell_contrib = np.sign(r_resp) * np.sqrt(r_resp**2 + r_non**2)
    return chi2, p, cell_contrib


def make_heatmap(per_event_tabs, alpha, out_path):
    """Two-panel heatmap: regions x events, colored by % responsive and by
    standardized residual on the `responsive` column.

    Cells where |residual| > 2 get an asterisk in the residual panel; cells
    in regions absent from a given event (filtered by --min-n) stay blank.
    """
    events  = list(per_event_tabs.keys())
    regions = sorted({r for tab in per_event_tabs.values() for r in tab.index})

    pct    = np.full((len(regions), len(events)), np.nan)
    resid  = np.full((len(regions), len(events)), np.nan)
    n_resp = np.zeros((len(regions), len(events)), dtype=int)
    n_tot  = np.zeros((len(regions), len(events)), dtype=int)
    headers = []

    chi2_stats = []
    for j, ev in enumerate(events):
        tab = per_event_tabs[ev]
        chi2, p, cell = _chi2_residuals(tab)
        stars = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
        headers.append(f"{ev}\np={p:.2g}{stars}")
        chi2_stats.append((chi2, p))

        idx_pos = {region: k for k, region in enumerate(tab.index)}
        for i, region in enumerate(regions):
            if region in idx_pos:
                k = idx_pos[region]
                pct[i, j]    = 100 * tab.loc[region, "responsive"] / tab.loc[region, "total"]
                resid[i, j]  = cell[k]
                n_resp[i, j] = tab.loc[region, "responsive"]
                n_tot[i, j]  = tab.loc[region, "total"]

    # Floor the width so the two panel titles never collide in the
    # single-event (variable-window) case.
    fig_w = max(7.0, 2.6 + 1.6 * len(events))
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 1.4 + 0.7 * len(regions)))

    # --- panel 1: % responsive ---
    ax = axes[0]
    im = ax.imshow(pct, cmap="viridis", aspect="auto", vmin=np.nanmin(pct), vmax=100)
    ax.set_title(f"% responsive (alpha={alpha})", fontsize=10)
    ax.set_xticks(range(len(events)));  ax.set_xticklabels(headers, fontsize=8)
    ax.set_yticks(range(len(regions))); ax.set_yticklabels(regions, fontsize=9)
    for i in range(len(regions)):
        for j in range(len(events)):
            if not np.isnan(pct[i, j]):
                ax.text(j, i, f"{pct[i, j]:.1f}%\n{n_resp[i, j]}/{n_tot[i, j]}",
                        ha="center", va="center", fontsize=7,
                        color="white" if pct[i, j] < 80 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="% responsive")

    # --- panel 2: signed sqrt of per-cell chi2 contribution ---
    # Magnitude = how much this cell adds to chi2; sign = direction on the
    # `responsive` column (+ = more responsive than chance, - = less).
    ax = axes[1]
    vmax = np.nanmax(np.abs(resid)) if np.isfinite(resid).any() else 1.0
    vmax = max(vmax, 2.0)
    im = ax.imshow(resid, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Signed sqrt(cell chi2 contribution)\n+ = more responsive than expected", fontsize=10)
    ax.set_xticks(range(len(events)));  ax.set_xticklabels(headers, fontsize=8)
    ax.set_yticks(range(len(regions))); ax.set_yticklabels(regions, fontsize=9)
    for i in range(len(regions)):
        for j in range(len(events)):
            if not np.isnan(resid[i, j]):
                mark = " *" if abs(resid[i, j]) > 2 else ""
                ax.text(j, i, f"{resid[i, j]:+.2f}{mark}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(resid[i, j]) > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label=r"signed $\sqrt{\Delta\chi^2}$")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def combined_any_event(events, alpha, window_end=None):
    """Per-neuron 'responsive to ANY of these events' across sessions."""
    pieces = []
    for event in events:
        df = load_event(event, window_end)
        if df is None:
            continue
        df = df[["session", "neuron_idx", "label", "p_zeta"]].copy()
        df["event"]      = event
        df["responsive"] = df["p_zeta"] < alpha
        pieces.append(df)
    if not pieces:
        return None
    long = pd.concat(pieces, ignore_index=True)
    agg = (long.groupby(["session", "neuron_idx", "label"])["responsive"]
                .any().reset_index())
    agg["region"] = agg["label"].map(parse_region)
    agg = agg.dropna(subset=["region"])
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--event", choices=list(EVENTS) + ["all"], default="all",
                   help="Event to test (default: each event separately + combined).")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance threshold on p_zeta (default 0.05).")
    p.add_argument("--min-n", type=int, default=5,
                   help="Drop regions with fewer than this many pooled neurons (default 5).")
    p.add_argument("--csv", action="store_true",
                   help="Save contingency tables to results/zeta_responsiveness/region_chisq_<event>.csv")
    p.add_argument("--plot", action="store_true",
                   help="Save a regions x events heatmap (% responsive + std. residuals).")
    p.add_argument("--window-end", default=None, choices=list(EVENTS),
                   help="Analyse the variable-window results "
                        "zeta_<event>_to_<window_end>_<session>.csv instead of "
                        "the fixed-window files. Requires a specific --event.")
    args = p.parse_args()

    if args.window_end and args.event == "all":
        p.error("--window-end requires a specific --event (e.g. --event cue --window-end reward).")
    if args.window_end == args.event:
        p.error("--window-end must differ from --event.")

    events  = list(EVENTS) if args.event == "all" else [args.event]
    out_dir = RESULTS_SUBDIRS["responsiveness"]
    # Tag used in printed labels, output filenames and heatmap keys.
    suffix  = f"_to_{args.window_end}" if args.window_end else ""

    def disp(event):
        return f"{event} -> {args.window_end}" if args.window_end else event

    per_event_tabs = {}  # collected for the heatmap

    for event in events:
        df = load_event(event, args.window_end)
        if df is None:
            print(f"\n=== {disp(event)} ===\n  no matching CSVs in {out_dir}")
            continue
        tab = contingency(df, args.alpha, args.min_n)
        run_test(tab, f"{disp(event)}  (alpha={args.alpha})")
        if not tab.empty:
            per_event_tabs[disp(event)] = tab
        if args.csv and not tab.empty:
            path = os.path.join(out_dir, f"region_chisq_{event}{suffix}.csv")
            tab.to_csv(path)
            print(f"  saved -> {path}")

    # The combined "ANY event" view only makes sense when several events are
    # pooled — skip it in the single-event variable-window case.
    if args.event == "all":
        agg = combined_any_event(events, args.alpha, args.window_end)
        if agg is not None:
            tab = (agg.groupby("region")["responsive"]
                      .value_counts().unstack(fill_value=0))
            for c in (True, False):
                if c not in tab.columns:
                    tab[c] = 0
            tab = tab.rename(columns={True: "responsive", False: "non_responsive"})
            tab = tab[["responsive", "non_responsive"]]
            tab["total"] = tab.sum(axis=1)
            tab = tab[tab["total"] >= args.min_n].sort_values("total", ascending=False)
            run_test(tab, f"ANY event  (alpha={args.alpha})")
            if not tab.empty:
                per_event_tabs["any"] = tab
            if args.csv and not tab.empty:
                path = os.path.join(out_dir, "region_chisq_any.csv")
                tab.to_csv(path)
                print(f"  saved -> {path}")

    if args.plot and per_event_tabs:
        name     = f"region_chisq_heatmap{suffix}.png" if suffix else "region_chisq_heatmap.png"
        out_path = os.path.join(out_dir, name)
        make_heatmap(per_event_tabs, args.alpha, out_path)


if __name__ == "__main__":
    main()
