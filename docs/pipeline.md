# pv-profiler – Pipeline (Block A–E)

> Ziel: **operationalisierbare** Beschreibung der Pipeline-Schritte und Artefakte.  
> Markierung: **[UNSICHER]** wenn alternative Varianten genannt wurden.

## Block A — Input & Zeitachse

### Inputs
- Zeitreihe(n) mindestens:
  - `ac_power_w` oder `power` (Spaltenname variiert; im Code vereinheitlichen)
  - `time` (Index oder Spalte)

### Schritte
1. CSV/Parquet laden.
2. Timestamp normalisieren:
   - Wenn tz-aware → beibehalten (aber DST-Policy beachten).
   - Wenn naive → lokalisieren mit Fixed Offset (Default `Etc/GMT-1`) oder CLI `--timezone`.
3. Resampling/Regularisierung auf 5‑min Raster (Anforderung im Projektkontext).
4. Grundreinigung:
   - NaNs markieren/entfernen gemäß Regeln (nicht vollständig ausformuliert).
   - Offsets/Zeitsprünge erkennen (Timeshift-Fix wurde diskutiert).

### Output
- `02_cleaned_timeshift_fixed.parquet`

## Block B — SDT Flags & Fit-Mask

### Ziel
Erzeuge konsistente Masks:
- `is_clipped_time`
- `is_clear_time` / `is_clear_day`
- `no_errors`
- `is_fit_time` / `is_fit_day`

### SDT APIs (rekonstruiert)
- `calculate_times(data, threshold=None, plot=False, ...)`
- `find_clipped_times()`
- `BooleanMasks` / `DailyFlags`

### Kritische Stelle: clear_times nicht zuverlässig
- `dh.boolean_masks.clear_times` wurde als häufig `None` beschrieben.
- Daher: **eigene Clear-Time-Logik**:
  - (a) Clear Days + zusätzliche Zeitfilterung innerhalb des Tages [UNSICHER]
  - (b) SDT `calculate_times()` Ergebnisse verwenden
  - (c) robustes Fallback, wenn SDT Mask fehlt

### Fit-Definition
- `is_fit_day = clear AND NOT inverter_clipped AND no_errors`
- optional: Fit auf Tagesebene ableiten und dann alle Tageszeiten übernehmen oder nur subset „clear times“.

### Output
- `03_clear_times_mask.parquet`
- `02_sdt_daily_flags.csv` (oder ähnlich)
- ggf. `05_power_fit.parquet` (normalisierte Fit-Zeitreihe)

## Block C — Forward-Modell & Zielprofil (MVP)

### Ziel
Erzeuge Modellprofile zur Orientierungsschätzung.

### Schritte
1. Normalisierung:
   - `p_norm = ac_power / daily_peak` oder äquivalent (Peak/Quantil; im Chat taucht `06_daily_peak.csv` auf).
2. Aggregation (MVP):
   - gruppiere nach `minute_of_day`
   - `target_profile[m] = median(p_norm at minute_of_day=m over fit days)`
3. Erzeuge Modellprofil(e) je Kandidat (tilt/az):
   - pvlib Clearsky (Ineichen u.a.)
   - POA via Transposition
   - Normierung auf vergleichbare Skala (z.B. max=1)

### Outputs
- `05_power_fit.parquet` (falls nicht schon Block B)
- Debug/Plot Artefakte [UNSICHER]
- `10_profile_compare.csv` (observed vs model profile)

## Block D — Orientation Search

### Single-plane (09a)
- Grid ranges:
  - tilt: (z. B. 0..60, step ?) [UNSICHER]
  - azimuth: (z. B. 90..270, step ?) [UNSICHER]
- Score:
  - RMSE (BIC optional zusätzlich)
- Output:
  - `09a_orientation_single_full_grid.csv`
  - `09_orientation_topk.csv` (Top‑k)

### Two-plane (09b)
- Parameterisierung (diskutiert):
  - Variante 1 (früh): `az_center` + `delta_az_deg = 90 (fix)` → später als falsch markiert.
  - Variante 2 (später): `az2 = az1 + 180` (Ost/West)
  - Zusatz: `weight_east` grid (0..1)
- Performance-Problem:
  - Full grid + weight grid → ~20 min Läufe.
  - Beschleunigung gefordert (siehe `design_decisions.md`).

### Output
- `09b_orientation_two_plane_full_grid.csv`
- `08_orientation_result.json`

## Block E — Residual / Shading

### Ziel
- Residuen zwischen Best-fit Modell und Observed analysieren.
- Shading-Indikatoren aus Residualstruktur ableiten.

### Status
- Ansatz diskutiert, aber Details nicht vollständig final. [UNSICHER]
- pv-analytics `features.shading` wurde als Option diskutiert, aber nicht übernommen. [UNSICHER]
