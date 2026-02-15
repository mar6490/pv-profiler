# pv-profiler

`pv-profiler` implementiert ein reproduzierbares, zweistufiges Identifikationsverfahren für PV-Anlagen auf Basis von AC-Leistungszeitreihen.

## Installation

```bash
pip install -e .
```

## Abhängigkeiten (wichtig)

- `solar-data-tools==2.1.1`
- `pvlib>=0.11.2,<0.12`
- `numpy<2.1`
- weitere Abhängigkeiten gemäß `pyproject.toml`

## Zeitkonzept (entscheidend)

- Interne Rechenzeitbasis ist **immer** `Etc/GMT-1` (fixed UTC+01:00, CET ohne DST).
- TZ-naive Eingaben werden deterministisch lokalisiert mit `tz_localize("Etc/GMT-1")`.
- TZ-aware Eingaben sind nur erlaubt, wenn sie fixed offset `+01:00` sind.
- `Europe/Berlin` (DST-aware) ist als Rechenzeitbasis nicht erlaubt.
- Optionale Anzeige-Konvertierungen (z. B. `Europe/Berlin`) sind nur für Reporting/Visualisierung gedacht.

## Input Contract

### Modus 1: Single-Plant File

- CSV/Parquet mit Spalten:
  - `timestamp`
  - `ac_power` (W, float, NaN erlaubt)
- Sampling-Regeln (strict):
  - exakt 5-Minuten-Raster
  - keine fehlenden Zeitpunkte
  - keine Duplikate
  - kein Resampling, keine automatische Korrektur

### Modus 2: Multi-Plant Wide File

- CSV/Parquet mit Spalten:
  - `timestamp`
  - weitere Spalten mit `system_id` als Spaltennamen
- Gleiche Sampling- und Zeitregeln wie oben.
- CLI splittet Wide-Format in einzelne Systeme.

### Standort/Metadaten

- Datei: `data/processed/plants.csv`
- Pflichtspalten: `system_id,country,plz,lat,lon,timezone`
- Standort-Lookup: `system_id -> (lat, lon)`
- Kein Geocoding / keine Online-APIs.
- Bei `run-single`: wenn kein Standort in Metadaten vorhanden ist, müssen `--lat --lon` angegeben werden.

## Pipeline-Blocks

### Block A (voll implementiert)

- A1 Parsing/Timezone-Validierung -> `01_parsed_tzaware.parquet`
- A2 SDT Onboarding
  - `DataHandler.run_pipeline(power_col="ac_power", fix_shifts=True, verbose=False)`
  - cleaned series extraction via `extract_clean_power_series(...)`
  - clear-times mask -> `03_clear_times_mask.parquet`
  - clipping mask (0.98 * Tagesmaximum während Daytime) -> `06_clipped_times_mask.parquet`
  - daily flags -> `04_daily_flags.csv`
  - summaries -> `05_clipping_summary.json`, `07_sdt_summary.json`, `07_sdt_introspect.json`
- A3 Exclusion flags in `summary.json`
  - `exclude_clipping`
  - `exclude_low_clear`
  - optional `suspect_large_shift`

### Block B (voll implementiert)

- `P_peak_day = quantile(0.995)` auf Daytime, ohne clipped points
- `p_norm = ac_power_clean / P_peak_day`
- Outputs:
  - `08_daily_peak.csv`
  - `09_p_norm.parquet`

### Block C (voll implementiert)

- `delta = abs(p_norm - p_norm.shift(1))`
- `is_smooth = delta <= tau`
- `fit_mask = is_clear_time & is_smooth & ~is_clipped_time`
- Outputs:
  - `11_fit_mask.parquet`
  - `12_daily_fit_fraction.csv`

### Block D/E (coming next)

- `orientation.py`, `capacity.py`, `shading.py` sind als dokumentierte Platzhalter vorhanden und werfen aktuell `NotImplementedError`.

## Output-Layout

Pro Anlage:

```text
outputs/<system_id>/<YYYYMMDD_HHMMSS>/
  01_parsed_tzaware.parquet
  02_cleaned_timeshift_fixed.parquet
  03_clear_times_mask.parquet
  04_daily_flags.csv
  05_clipping_summary.json
  06_clipped_times_mask.parquet
  07_sdt_summary.json
  07_sdt_introspect.json
  08_daily_peak.csv
  09_p_norm.parquet
  11_fit_mask.parquet
  12_daily_fit_fraction.csv
  summary.json
```

Aggregierter Report:

```text
outputs/reports/report_summary.csv
outputs/reports/report_summary.json
```

## CLI

```bash
pv-ident run-single -c examples/config.yml --system-id sys_demo --input examples/example_single.csv
pv-ident run-wide -c examples/config.yml --input examples/example_wide.csv
pv-ident run -c examples/config.yml --manifest examples/manifest.csv
pv-ident report -c examples/config.yml --output-root outputs
```
