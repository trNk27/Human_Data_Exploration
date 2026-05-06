"""Convert all session .mat files to CSV under csv/<session>/<file>.csv."""

import os
import re
import scipy.io
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_ROOT = os.path.join(BASE_DIR, "csv")

TRIALS_SYNC_COLS = [
    "TrialStart_s", "TrialEnd_s", "TrialDuration_s", "Block",
    "GambleSide_R1L0", "P_BigReward_Gamble", "P_SmallReward_Safe",
    "Amount_BigReward_Gamble", "Amount_SmallReward_Safe",
    "PriorWheelNotStopping", "NotResponding",
    "ChosenSide_unreliable", "ChosenArm_G1S0", "Rewarded",
    "TrialStart_sp", "CuePresent_sp", "RespWindowStart_sp",
    "RewardOnset_sp", "TrialEnd_sp",
]


def session_dirs():
    """Yield (name, path) for every directory that contains at least one .mat file."""
    for entry in sorted(os.listdir(BASE_DIR)):
        path = os.path.join(BASE_DIR, entry)
        if os.path.isdir(path) and any(f.endswith(".mat") for f in os.listdir(path)):
            yield entry, path


def convert_sr(data):
    sr = int(np.asarray(data["SR"]).flat[0])
    return {"SR": pd.DataFrame({"SamplingRate_Hz": [sr]})}


def convert_stmtx(data):
    matrix = data["STMtx"]
    if "infoCell" in data:
        info = data["infoCell"]

        def clean(cell):
            return str(cell).strip().strip("[]'\"")

        cols = [
            f"{clean(info[i, 2])} | {clean(info[i, 0])} {clean(info[i, 1])} ({clean(info[i, 3])})"
            for i in range(info.shape[0])
        ]
    else:
        cols = [f"unit_{i}" for i in range(matrix.shape[1])]
    return {"STMtx": pd.DataFrame(matrix, columns=cols)}


def convert_trials_sync(data):
    matrix = np.atleast_2d(data["Trials_Sync"])
    n_cols = matrix.shape[1]
    cols = TRIALS_SYNC_COLS[:n_cols]
    if n_cols > len(TRIALS_SYNC_COLS):
        cols += [f"col_{i}" for i in range(len(TRIALS_SYNC_COLS), n_cols)]
    return {"Trials_Sync": pd.DataFrame(matrix, columns=cols)}


def mat_struct_to_df(obj):
    """Best-effort conversion of a scipy mat_struct or structured ndarray to a DataFrame."""
    if hasattr(obj, "_fieldnames"):
        # mat_struct (from squeeze_me=True)
        rows = {}
        for field in obj._fieldnames:
            val = getattr(obj, field)
            if isinstance(val, np.ndarray) and val.ndim <= 1:
                rows[field] = val.flatten()
            elif np.isscalar(val):
                rows[field] = [val]
        return pd.DataFrame(rows)
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        return pd.DataFrame({name: obj[name].flatten() for name in obj.dtype.names})
    return None


def _extract_df_from_value(val):
    """Recursively try to pull a useful DataFrame out of a scipy-loaded mat value."""
    if isinstance(val, np.ndarray):
        if val.dtype.names:
            return pd.DataFrame({name: val[name].flatten() for name in val.dtype.names})
        if val.ndim == 2 and not val.dtype == object:
            return pd.DataFrame(val)
        if val.dtype == object and val.size == 1:
            return _extract_df_from_value(val.flat[0])
    if hasattr(val, "_fieldnames"):
        # mat_struct — may be a MATLAB table wrapper; look for a "data" field first
        fields = val._fieldnames
        if "data" in fields and "colnames" in fields:
            raw = getattr(val, "data")
            cols = getattr(val, "colnames")
            if isinstance(cols, np.ndarray):
                cols = [str(c).strip() for c in cols.flatten()]
            if isinstance(raw, np.ndarray) and raw.dtype.names:
                return pd.DataFrame({name: raw[name].flatten() for name in raw.dtype.names})
            if isinstance(raw, np.ndarray) and raw.ndim == 2:
                return pd.DataFrame(raw, columns=cols if len(cols) == raw.shape[1] else None)
        # Generic struct → one column per field
        rows = {}
        for field in fields:
            v = getattr(val, field)
            arr = np.asarray(v).flatten()
            rows[field] = arr
        lengths = {len(v) for v in rows.values()}
        if len(lengths) == 1:
            return pd.DataFrame(rows)
    return None


def convert_human_data_table(data):
    # Re-load without squeeze_me so arrays keep their shape, allowing dtype.names to survive
    mat_path = data["__file__"]
    try:
        raw = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=False)
    except Exception:
        raw = data

    frames = {}
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        df = _extract_df_from_value(val)
        if df is not None and not df.empty:
            frames[key] = df
    return frames


CONVERTERS = {
    "SR.mat": convert_sr,
    "STMtx.mat": convert_stmtx,
    "Trials_Sync.mat": convert_trials_sync,
    "Human_Data_Table.mat": convert_human_data_table,
}


def convert_session(session_name, session_path):
    out_dir = os.path.join(CSV_ROOT, session_name)
    os.makedirs(out_dir, exist_ok=True)
    mat_files = [f for f in os.listdir(session_path) if f.endswith(".mat")]

    for fname in sorted(mat_files):
        mat_path = os.path.join(session_path, fname)
        try:
            data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            data["__file__"] = mat_path
        except Exception as e:
            print(f"  [SKIP] {fname}: could not load ({e})")
            continue

        converter = CONVERTERS.get(fname)
        if converter is None:
            print(f"  [SKIP] {fname}: no converter registered")
            continue

        try:
            frames = converter(data)
        except Exception as e:
            print(f"  [ERROR] {fname}: conversion failed ({e})")
            continue

        for label, df in frames.items():
            stem = os.path.splitext(fname)[0]
            csv_name = f"{stem}.csv" if label == stem else f"{stem}_{label}.csv"
            csv_path = os.path.join(out_dir, csv_name)
            df.to_csv(csv_path, index=False)
            print(f"  {csv_name}  ({df.shape[0]} rows x {df.shape[1]} cols)")


def main():
    print(f"Output root: {CSV_ROOT}\n")
    for name, path in session_dirs():
        print(f"Session: {name}")
        convert_session(name, path)
        print()


if __name__ == "__main__":
    main()
