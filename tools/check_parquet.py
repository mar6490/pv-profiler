import pandas as pd
from pathlib import Path

INPUT_PATH = Path(
    r"C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\data\sonnja_pv3_2015\einleuchtend_wrdata_2015_wr1_5min.parquet"
)

def main():
    print("Reading parquet...")
    df = pd.read_parquet(INPUT_PATH)

    print("\n--- BASIC INFO ---")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Index type:", type(df.index))

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not DatetimeIndex.")

    print("Timezone:", df.index.tz)
    print("Monotonic increasing:", df.index.is_monotonic_increasing)
    print("Has duplicates:", df.index.has_duplicates)

    # Frequency
    inferred_freq = pd.infer_freq(df.index[:5000])
    print("Inferred frequency (first 5000):", inferred_freq)

    # Strict 5 min check
    diffs = df.index.to_series().diff().dropna().unique()
    print("Unique time deltas:", diffs[:5])
    print("Number of unique deltas:", len(diffs))

    if len(diffs) != 1:
        raise ValueError("Time deltas not constant.")

    if diffs[0] != pd.Timedelta("5min"):
        raise ValueError("Frequency is not exactly 5 minutes.")

    print("\n--- NaN CHECK ---")
    print(df.isna().sum())

    print("\n✔ File satisfies structural requirements.")

if __name__ == "__main__":
    main()
