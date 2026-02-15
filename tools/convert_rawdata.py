import pandas as pd
from pathlib import Path

# ========= CONFIG =========
INPUT_PATH = Path(
    r"C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\data\sonnja_pv3_2015\einleuchtend_wrdata_2015_wr1.csv"
)
OUTPUT_PATH = INPUT_PATH.with_name("einleuchtend_wrdata_2015_wr1_5min.parquet")

SEP = ";"
TIMESTAMP_COL = "timestamp"
POWER_COL = "P_AC"

TARGET_FREQ = "5min"
ACCEPTED_1MIN_FREQS = {"T", "min", "1min", "1T"}  # pandas may return different aliases
# ==========================


def main():
    print("Reading file (only needed columns)...")
    df = pd.read_csv(
        INPUT_PATH,
        sep=SEP,
        usecols=[TIMESTAMP_COL, POWER_COL],
        low_memory=False,
    )

    print("Parsing timestamps...")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="raise")

    print("Sorting by timestamp...")
    df = df.sort_values(TIMESTAMP_COL)

    print("Checking duplicates...")
    if df[TIMESTAMP_COL].duplicated().any():
        dups = df.loc[df[TIMESTAMP_COL].duplicated(), TIMESTAMP_COL].head(10).tolist()
        raise ValueError(f"Duplicate timestamps detected (examples): {dups}")

    df = df.set_index(TIMESTAMP_COL)

    print("Checking monotonic order...")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamps not sorted after sorting step (unexpected).")

    print("Inferring original frequency from first 2000 points...")
    inferred_freq = pd.infer_freq(df.index[:2000])
    print("Inferred frequency:", inferred_freq)

    if inferred_freq is None:
        raise ValueError(
            "Could not infer a regular frequency (infer_freq returned None). "
            "This usually indicates gaps or irregular sampling."
        )

    if inferred_freq not in ACCEPTED_1MIN_FREQS:
        raise ValueError(
            f"Expected ~1-minute frequency, but got {inferred_freq}. "
            "If the series has gaps, you must decide how to handle them."
        )

    print("Coercing power to float...")
    power = pd.to_numeric(df[POWER_COL], errors="coerce")

    print(f"Resampling to {TARGET_FREQ} mean...")
    power_5min = power.resample(TARGET_FREQ).mean()

    print("Checking resulting frequency...")
    inferred_5min = pd.infer_freq(power_5min.index[:2000])
    print("Inferred new frequency:", inferred_5min)

    if inferred_5min != TARGET_FREQ:
        raise ValueError(f"Resampled frequency mismatch. Expected {TARGET_FREQ}, got {inferred_5min}")

    print("Checking missing 5-min bins...")
    full_range = pd.date_range(
        start=power_5min.index.min(),
        end=power_5min.index.max(),
        freq=TARGET_FREQ,
    )
    if len(full_range) != len(power_5min):
        # show a couple missing bins to help debugging
        missing = full_range.difference(power_5min.index)
        examples = list(missing[:10])
        raise ValueError(
            f"Missing timestamps after resampling (gaps in 5-min bins). "
            f"Examples: {examples}"
        )

    out = power_5min.to_frame(name=POWER_COL).reset_index()

    print("Saving parquet...")
    out.to_parquet(OUTPUT_PATH, index=False)

    print("Done.")
    print(f"Original rows: {len(df)}")
    print(f"Resampled rows: {len(out)}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
