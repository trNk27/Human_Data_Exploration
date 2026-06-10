"""Build a per-session, per-neuron summary CSV.

Discovers all ZETA result CSVs for each session, joins them on neuron_idx,
extracts metadata from the label string, and writes a single tidy table:

    results/neuron_summary_<session>.csv

One row per neuron.  Column groups:

  Identity
    session           recording session ID
    neuron_idx        integer column index in STMtx.mat
    label             full label string  (unit<id> | <REGION> ele<elec> (<type>))
    unit_id           integer unit ID parsed from label
    region            brain region  (e.g. SMG, AG, IFG, MFG)
    electrode         electrode ID  (e.g. 082)
    unit_type         "su" (single unit) or "mu" (multi unit)

  ZETA responsiveness  (suffix = event name: cue / reward / trial_start / cue_to_reward)
    p_zeta_<event>          p-value from one-sample ZETA test
    zeta_<event>            ZETA statistic
    latency_<event>_s       latency of response in seconds
    peak_onset_<event>_s    peak-onset time in seconds

  ZETA outcome  (suffix = contrast: rew_outcome / choice_outcome)
    p_zeta2_<contrast>      p-value from two-sample ZETA test
    zeta2_<contrast>        ZETA statistic
    zeta2_t_<contrast>_s    time of maximum condition difference (s)
    rate_GR_<contrast>      mean firing rate, G+Rewarded trials  (Hz)
    rate_GN_rew_outcome     mean firing rate, G+Non-rewarded trials  (Hz)
    rate_SR_choice_outcome  mean firing rate, S+Rewarded trials  (Hz)
    SI_<contrast>           selectivity index = (rate_A − rate_B) / (rate_A + rate_B)
    pref_<contrast>         preference label ("rewarded"/"non-rewarded" or "gamble"/"safe")

Usage
-----
    python -m analysis.build_neuron_table              # all sessions
    python -m analysis.build_neuron_table --session 20250714
    python analysis/build_neuron_table.py --session 20250714
"""

import argparse
import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap (so the script works both as a module and directly)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import REPO_ROOT, RESULTS_DIR, RESULTS_SUBDIRS  # noqa: E402

SUMMARY_DIR = RESULTS_DIR  # output goes straight into results/


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(
    r"unit(\d+)\s*\|\s*(\S+)\s+ele(\w+)\s*\((\w+)\)"
)


def parse_label(label: str) -> dict:
    """Return dict with unit_id, region, electrode, unit_type — or empty strings on failure."""
    m = _LABEL_RE.search(str(label))
    if m:
        return {
            "unit_id":   int(m.group(1)),
            "region":    m.group(2),
            "electrode": m.group(3),
            "unit_type": m.group(4),
        }
    return {"unit_id": None, "region": None, "electrode": None, "unit_type": None}


# ---------------------------------------------------------------------------
# File-pattern descriptors
# ---------------------------------------------------------------------------

# Each entry:  (glob_stem, output_suffix, column_renames)
# glob_stem:    the part between "zeta_" / "zeta2_" and "_<session>.csv"
# output_suffix: appended to each result column name
# column_renames: original_col -> new_col  (neuron_idx and label are handled separately)

RESPONSIVENESS_SOURCES = [
    (
        "cue",
        "_cue",
        {
            "p_zeta":        "p_zeta_cue",
            "zeta":          "zeta_cue",
            "latency_s":     "latency_cue_s",
            "peak_onset_s":  "peak_onset_cue_s",
        },
    ),
    (
        "reward",
        "_reward",
        {
            "p_zeta":        "p_zeta_reward",
            "zeta":          "zeta_reward",
            "latency_s":     "latency_reward_s",
            "peak_onset_s":  "peak_onset_reward_s",
        },
    ),
    (
        "trial_start",
        "_trial_start",
        {
            "p_zeta":        "p_zeta_trial_start",
            "zeta":          "zeta_trial_start",
            "latency_s":     "latency_trial_start_s",
            "peak_onset_s":  "peak_onset_trial_start_s",
        },
    ),
    (
        "cue_to_reward",
        "_cue_to_reward",
        {
            "p_zeta":        "p_zeta_cue_to_reward",
            "zeta":          "zeta_cue_to_reward",
            "latency_s":     "latency_cue_to_reward_s",
            "peak_onset_s":  "peak_onset_cue_to_reward_s",
        },
    ),
]

OUTCOME_SOURCES = [
    (
        "reward",       # zeta2_reward_<session>.csv  — G+R vs G+N
        "_rew_outcome",
        {
            "p_zeta":    "p_zeta2_rew_outcome",
            "zeta":      "zeta2_rew_outcome",
            "zeta_t_s":  "zeta2_t_rew_outcome_s",
            "rate_GR":   "rate_GR_rew_outcome",
            "rate_GN":   "rate_GN_rew_outcome",
            "SI":        "SI_rew_outcome",
            "preference":"pref_rew_outcome",
        },
    ),
    (
        "choice",       # zeta2_choice_<session>.csv  — G+R vs S+R
        "_choice_outcome",
        {
            "p_zeta":    "p_zeta2_choice_outcome",
            "zeta":      "zeta2_choice_outcome",
            "zeta_t_s":  "zeta2_t_choice_outcome_s",
            "rate_GR":   "rate_GR_choice_outcome",
            "rate_SR":   "rate_SR_choice_outcome",
            "SI":        "SI_choice_outcome",
            "preference":"pref_choice_outcome",
        },
    ),
]


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _load_and_rename(path: str, col_map: dict) -> pd.DataFrame | None:
    """Load a CSV, keep neuron_idx + mapped columns, return None if file missing."""
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    keep = {"neuron_idx": "neuron_idx"}
    keep.update({k: v for k, v in col_map.items() if k in df.columns})
    return df[list(keep.keys())].rename(columns=keep)


def build_session_table(session: str) -> pd.DataFrame:
    resp_dir    = RESULTS_SUBDIRS["responsiveness"]
    outcome_dir = RESULTS_SUBDIRS["outcome"]

    frames: list[pd.DataFrame] = []

    # ---- responsiveness ----
    for stem, _suffix, col_map in RESPONSIVENESS_SOURCES:
        path = os.path.join(resp_dir, f"zeta_{stem}_{session}.csv")
        chunk = _load_and_rename(path, col_map)
        if chunk is not None:
            frames.append(chunk)

    # ---- outcome ----
    for stem, _suffix, col_map in OUTCOME_SOURCES:
        path = os.path.join(outcome_dir, f"zeta2_{stem}_{session}.csv")
        chunk = _load_and_rename(path, col_map)
        if chunk is not None:
            frames.append(chunk)

    if not frames:
        return pd.DataFrame()

    # Merge all chunks on neuron_idx (outer so no neuron is dropped)
    result = frames[0]
    for chunk in frames[1:]:
        result = result.merge(chunk, on="neuron_idx", how="outer")

    # ---- label: pick the first non-null value across all source files ----
    label_frames = []
    for stem, _suffix, _col_map in RESPONSIVENESS_SOURCES + [("reward", "", {}), ("choice", "", {})]:
        is_outcome = _suffix == ""  # outcome sources passed with empty suffix above
        if is_outcome:
            path = os.path.join(outcome_dir, f"zeta2_{stem}_{session}.csv")
        else:
            path = os.path.join(resp_dir, f"zeta_{stem}_{session}.csv")
        if os.path.isfile(path):
            tmp = pd.read_csv(path)[["neuron_idx", "label"]]
            label_frames.append(tmp)
    if label_frames:
        label_df = pd.concat(label_frames, ignore_index=True).drop_duplicates("neuron_idx")
        result = result.merge(label_df, on="neuron_idx", how="left")
    else:
        result["label"] = None

    # ---- parse label into metadata columns ----
    parsed = result["label"].apply(parse_label).apply(pd.Series)
    result = pd.concat([result, parsed], axis=1)

    # ---- add session column and reorder ----
    result.insert(0, "session", session)

    identity_cols = ["session", "neuron_idx", "label", "unit_id", "region", "electrode", "unit_type"]
    other_cols    = [c for c in result.columns if c not in identity_cols]
    result = result[identity_cols + sorted(other_cols)]

    result = result.sort_values("neuron_idx").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_sessions() -> list[str]:
    """Return all session IDs that have at least one ZETA result CSV."""
    sessions = set()
    for d in RESULTS_SUBDIRS.values():
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            m = re.search(r"(\d{8})\.csv$", fname)
            if m:
                sessions.add(m.group(1))
    return sorted(sessions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", nargs="+", metavar="SESSION",
                        help="Session ID(s) to process (default: all discovered sessions)")
    args = parser.parse_args()

    sessions = args.session if args.session else discover_sessions()
    if not sessions:
        print("No sessions found — run the ZETA analyses first.")
        return

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    for session in sessions:
        print(f"[{session}]  building neuron table …", end="  ", flush=True)
        df = build_session_table(session)
        if df.empty:
            print("no data found, skipping.")
            continue
        out_path = os.path.join(SUMMARY_DIR, f"neuron_summary_{session}.csv")
        df.to_csv(out_path, index=False)
        print(f"{len(df)} neurons -> {os.path.relpath(out_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
