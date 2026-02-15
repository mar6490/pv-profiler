import pandas as pd
from pathlib import Path

IN_PATH = Path(r"C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\data\sonnja_pv3_2015\einleuchtend_wrdata_2015_wr1_5min.parquet")
OUT_PATH = IN_PATH.with_name("einleuchtend_wrdata_2015_wr1_5min_tool.parquet")

TIMESTAMP_COL = "timestamp"
POWER_COL = "P_AC"
TZ = "Etc/GMT-1"  # fixed UTC+01:00, no DST

def main():
    df = pd.read_parquet(IN_PATH)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"Missing '{TIMESTAMP_COL}' column.")
    if POWER_COL not in df.columns:
        raise ValueError(f"Missing '{POWER_COL}' column.")

    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="raise")

    # enforce tz-aware fixed-offset (naive -> localize)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(TZ)
    else:
        ts = ts.dt.tz_convert(TZ)

    out = df[[POWER_COL]].copy()
    out.index = pd.DatetimeIndex(ts, name="timestamp")
    out = out.sort_index()

    if out.index.has_duplicates:
        raise ValueError("Duplicate timestamps after setting index.")

    diffs = out.index.to_series().diff().dropna().unique()
    if len(diffs) != 1 or diffs[0] != pd.Timedelta("5min"):
        raise ValueError(f"Not strict 5-min sampling. Unique deltas: {diffs[:10]}")

    out.to_parquet(OUT_PATH)
    print("Wrote:", OUT_PATH)
    print("Index tz:", out.index.tz)
    print("Rows:", len(out), "Cols:", list(out.columns))
    print(out.head(3))

if __name__ == "__main__":
    main()
