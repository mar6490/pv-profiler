import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


PARQUET_PATH = r"C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\outputs\sonnja_wr1_block4_norm\07_p_norm_clear.parquet"
COL = "p_norm"  # ggf. anpassen

df = pd.read_parquet(PARQUET_PATH)



print(df.head(10))
print(df.dtypes)
print(df.index[:3])
print(df.index.tz)

# Anzahl Zeilen
print("Rows:", len(df))

# NaN pro Spalte
print("\nNaN per column:")
print(df.isna().sum())

# Gesamtanzahl NaN im DataFrame
print("\nTotal NaNs:", df.isna().sum().sum())

flags = pd.read_csv(r"C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\outputs\sonnja_wr1_block2_from_csv\02_sdt_daily_flags.csv", index_col=0, parse_dates=True)
fit_days = flags[(flags["clear"]) & (~flags["inverter_clipped"]) & (flags["no_errors"])]

print("Fit days count:", len(fit_days))
print(fit_days.index.month.value_counts().sort_index())

# Sicherstellen: Index ist datetime
if not isinstance(df.index, pd.DatetimeIndex):
    raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")



# nur Fit-Samples (NaNs raus) -> dann sind es nur Fit-Tage
s = df[COL].dropna()

# --- 1) Spaghetti-Plot: alle Fit-Tage überlagert (x = Uhrzeit) ---
# Minuten seit Mitternacht als "time-of-day" Achse
tod_min = s.index.hour * 60 + s.index.minute
dates = s.index.date

# (date, tod_min) -> power Matrix
mat = (
    pd.DataFrame({"date": dates, "tod_min": tod_min, "power": s.values}, index=s.index)
    .pivot_table(index="date", columns="tod_min", values="power", aggfunc="mean")
    .sort_index()
)

x = mat.columns.values / 60.0  # Stunden 0..24

plt.figure(figsize=(14, 5))
for _, row in mat.iterrows():
    plt.plot(x, row.values, linewidth=0.7, alpha=0.35)
plt.title("Daily power profiles (fit days overlaid)")
plt.xlabel("Hour of day")
plt.ylabel("Power")
plt.xlim(0, 24)
plt.tight_layout()
plt.show()

# Optional: Median/Quantile-Band darüberlegen (hilft bei Beurteilung)
p50 = np.nanmedian(mat.values, axis=0)
p10 = np.nanpercentile(mat.values, 10, axis=0)
p90 = np.nanpercentile(mat.values, 90, axis=0)

plt.figure(figsize=(14, 5))
plt.plot(x, p50, linewidth=2.0)
plt.fill_between(x, p10, p90, alpha=0.2)
plt.title("Fit-day profile summary (median with 10–90% band)")
plt.xlabel("Hour of day")
plt.ylabel("Power")
plt.xlim(0, 24)
plt.tight_layout()
plt.show()

# --- 2) Heatmap: Datum vs Uhrzeit (Power als Farbe) ---
plt.figure(figsize=(14, 7))
plt.imshow(mat.values, aspect="auto", origin="lower",
           extent=[x.min(), x.max(), 0, len(mat.index)])
plt.title("Heatmap of fit-day power (rows=days, cols=time-of-day)")
plt.xlabel("Hour of day")
plt.ylabel("Fit day index (sorted by date)")
plt.tight_layout()
plt.show()


# --- Plot 1: Full-year time series (NaNs => non-fit days) ---
plt.figure(figsize=(14, 5))
plt.plot(df.index, df[COL], linewidth=0.5)
plt.title("Power (fit-masked) over full year (NaNs = excluded days)")
plt.xlabel("Time")
plt.ylabel("Power")
plt.tight_layout()
plt.show()

# --- Plot 2: Fit-day distribution via daily peak ---
# daily peak on fit-masked series (NaNs ignored)
daily_peak_fit = df[COL].resample("D").max()

# mark which days had any fit sample (i.e., not all-NaN)
has_fit_day = daily_peak_fit.notna()

plt.figure(figsize=(14, 4))
plt.plot(daily_peak_fit.index, daily_peak_fit.values, marker=".", linestyle="None")
plt.title("Daily peak on fit-masked series (shows distribution of fit days)")
plt.xlabel("Date")
plt.ylabel("Daily peak power")
plt.tight_layout()
plt.show()

# Optional: "rug plot" style (fit days as 1/0 line)
plt.figure(figsize=(14, 1.8))
plt.step(has_fit_day.index, has_fit_day.astype(int), where="mid")
plt.ylim(-0.1, 1.1)
plt.title("Fit days over the year (1 = fit day, 0 = excluded)")
plt.xlabel("Date")
plt.yticks([0, 1], ["excluded", "fit"])
plt.tight_layout()
plt.show()


flags = pd.read_csv(r"C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\outputs\sonnja_wr1_block2_from_csv\02_sdt_daily_flags.csv", index_col=0, parse_dates=True)
flags["is_fit_day"] = (flags["clear"] == True) & (flags["inverter_clipped"] == False) & (flags["no_errors"] == True)

print(flags.loc[flags["is_fit_day"], ["cloudy", "density", "linearity"]].mean())

