import os
import scipy.io
import numpy as np
import pandas as pd

SESSION = "20250707"
DATA_DIR = os.path.join(os.path.dirname(__file__), SESSION)

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)


def load_sr(data_dir=None):
    data_dir = data_dir or DATA_DIR
    data = scipy.io.loadmat(os.path.join(data_dir, "SR.mat"))
    sr = int(data["SR"].flat[0])
    return pd.DataFrame({"SamplingRate_Hz": [sr]})


def load_stmtx(data_dir=None):
    data_dir = data_dir or DATA_DIR
    data = scipy.io.loadmat(os.path.join(data_dir, "STMtx.mat"))
    matrix = data["STMtx"]          # (samples, units)
    info = data["infoCell"]         # (units, 4): area, electrode, unit, type

    def clean(cell):
        s = str(cell).strip()
        return s.strip("[]'\"")

    cols = [
        f"{clean(info[i,2])} | {clean(info[i,0])} {clean(info[i,1])} ({clean(info[i,3])})"
        for i in range(info.shape[0])
    ]
    return pd.DataFrame(matrix, columns=cols)


def load_trials_sync(data_dir=None):
    data_dir = data_dir or DATA_DIR
    data = scipy.io.loadmat(os.path.join(data_dir, "Trials_Sync.mat"))
    matrix = data["Trials_Sync"]    # (trials, 19)
    # Columns 1-14 are behavioural; 15-19 are timing in sampling points (divide by SR for seconds).
    # ChosenSide (col 12) is unreliable per README — use ChosenArm (col 13) instead.
    col_names = [
        "TrialStart_s", "TrialEnd_s", "TrialDuration_s", "Block",
        "GambleSide_R1L0", "P_BigReward_Gamble", "P_SmallReward_Safe",
        "Amount_BigReward_Gamble", "Amount_SmallReward_Safe",
        "PriorWheelNotStopping", "NotResponding",
        "ChosenSide_unreliable", "ChosenArm_G1S0", "Rewarded",
        "TrialStart_sp", "CuePresent_sp", "RespWindowStart_sp",
        "RewardOnset_sp", "TrialEnd_sp",
    ]
    return pd.DataFrame(matrix, columns=col_names)


def main():
    sections = {
        "SR.mat - Sampling Rate": load_sr,
        "STMtx.mat - Spike Train Matrix (first 10 rows, first 8 units)": lambda: load_stmtx().iloc[:10, :8],
        "Trials_Sync.mat - Trial Data (first 20 trials)": lambda: load_trials_sync().head(20),
    }

    for title, loader in sections.items():
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
        df = loader()
        print(df.to_string(index=True))
        full = loader.__wrapped__() if hasattr(loader, "__wrapped__") else None

    # Full shape info
    stm = load_stmtx()
    ts = load_trials_sync()
    print(f"\n{'='*80}")
    print("  Summary")
    print(f"{'='*80}")
    print(f"  STMtx   : {stm.shape[0]} samples x {stm.shape[1]} units")
    print(f"  Trials  : {ts.shape[0]} trials x {ts.shape[1]} columns")
    sr_val = load_sr()["SamplingRate_Hz"].iloc[0]
    print(f"  SR      : {sr_val} Hz  =>  {stm.shape[0]/sr_val:.1f} s of recording")


if __name__ == "__main__":
    main()
