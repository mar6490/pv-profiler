import pandas as pd

inp = r"C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\data\2026-02-21_17-15-58\system_001.csv"
out = r"C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\data\2026-02-21_17-15-58\naiv\system_001_naive.csv"

df = pd.read_csv(inp)
ts = pd.to_datetime(df["time"])
# tz-aware -> tz-naive, aber KEINE Uhrzeitverschiebung (Offset wird nur “vergessen”)
df["time"] = ts.dt.tz_localize(None)
df.to_csv(out, index=False)
print("Wrote:", out)
