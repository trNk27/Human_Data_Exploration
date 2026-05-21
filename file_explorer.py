"""Print a quick textual overview of one session's .mat files."""

import argparse

import pandas as pd

from utils import (
    add_session_arg, session_data_dir,
    load_sr, load_stmtx, load_trials_sync,
)

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)


def main():
    parser = argparse.ArgumentParser(description="Inspect a session's .mat files.")
    add_session_arg(parser)
    args = parser.parse_args()
    data_dir = session_data_dir(args.session)

    print(f"Session: {args.session}\n")

    sections = {
        "SR.mat - Sampling Rate":
            lambda: load_sr(data_dir=data_dir),
        "STMtx.mat - Spike Train Matrix (first 10 rows, first 8 units)":
            lambda: load_stmtx(data_dir=data_dir).iloc[:10, :8],
        "Trials_Sync.mat - Trial Data (first 20 trials)":
            lambda: load_trials_sync(data_dir=data_dir).head(20),
    }

    for title, loader in sections.items():
        print(f"\n{'='*80}\n  {title}\n{'='*80}")
        print(loader().to_string(index=True))

    stm = load_stmtx(data_dir=data_dir)
    ts  = load_trials_sync(data_dir=data_dir)
    sr_val = load_sr(data_dir=data_dir)["SamplingRate_Hz"].iloc[0]
    print(f"\n{'='*80}\n  Summary\n{'='*80}")
    print(f"  STMtx   : {stm.shape[0]} samples x {stm.shape[1]} units")
    print(f"  Trials  : {ts.shape[0]} trials x {ts.shape[1]} columns")
    print(f"  SR      : {sr_val} Hz  =>  {stm.shape[0] / sr_val:.1f} s of recording")


if __name__ == "__main__":
    main()
