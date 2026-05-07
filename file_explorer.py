import pandas as pd

from utils import SESSION, DATA_DIR, load_sr, load_stmtx, load_trials_sync

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)


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
