# pv-profiler

`pv-profiler` implementiert ein reproduzierbares, zweistufiges Identifikationsverfahren für PV-Anlagen auf Basis von AC-Leistungszeitreihen.

## Installation

```bash
pip install -e .
```

SDT-Hinweis:

- Standardmäßig wird `solar-data-tools` aus PyPI installiert (aktuell typischerweise 2.1.1).
- Optional kann SDT v2.1.2 direkt vom Git-Tag installiert werden:

```bash
pip install "solar-data-tools @ git+https://github.com/slacgismo/solar-data-tools.git@v2.1.2"
```

## Abhängigkeiten (wichtig)

- `solar-data-tools>=2.1.1,<2.2`
- `pvlib>=0.11.2,<0.12`
- `numpy<2.1`
- weitere Abhängigkeiten gemäß `pyproject.toml`

## Zeitkonzept (entscheidend)

- Interne Rechenzeitbasis ist **immer** `Etc/GMT-1` (fixed UTC+01:00, CET ohne DST).
- TZ-naive Eingaben werden deterministisch lokalisiert mit `tz_localize("Etc/GMT-1")`.
- TZ-aware Eingaben sind nur erlaubt, wenn sie fixed offset `+01:00` sind.
- `Europe/Berlin` (DST-aware) ist als Rechenzeitbasis nicht erlaubt.
- Optionale Anzeige-Konvertierungen sind nur für Export/Plots gedacht.

## Input Contract

### Modus 1: Single-Plant File

- CSV/Parquet mit Spalten:
  - `timestamp` (konfigurierbar via `input.timestamp_col`)
  - `ac_power` oder z. B. `P_AC` (konfigurierbar via `input.power_col`)
  - alternativ darf bei Parquet der Timestamp als `DatetimeIndex` vorliegen
- Sampling-Regeln (strict):
  - exakt 5-Minuten-Raster
  - keine fehlenden Zeitpunkte
  - keine Duplikate
  - kein Resampling, keine automatische Korrektur

### Modus 2: Multi-Plant Wide File

- CSV/Parquet mit `timestamp` + mehrere Systemspalten (`system_id` als Spaltenname)
- gleiche Sampling-/Zeitregeln wie oben

### Standort/Metadaten

- Primär: `data/processed/plants.csv` mit `system_id, lat, lon`
- `run-single` unterstützt zusätzlich:
  - `--metadata-json <path>` (liest lat/lon, optional altitude)
  - fallback auf `--lat --lon`

## Pipeline

### Block A–C (bestehend)

- A: SDT Onboarding, Clear-Day-Detection, fit-times/clipping masks, A3 Exclusion-Flags
  - `pipeline.skip_clipping: true` (default) überspringt SDT-Clipping-Detection aus Stabilitätsgründen und nutzt eine no-clipping-Maske.
- B: Normalisierung mit Q0.995
- C: Fit-Maske (`fit_mask`)

### Block D (voll implementiert)

- Orientation Grid Search (tilt 0..60°) mit Single-Plane und optionalem East-West Two-Plane Modell
- Single-Plane: Azimuth-Suche im Bereich `orientation.az_min..az_max`
- Two-Plane: feste Azimuths E/W (90/270), Tilt-Suche + Modellvergleich über relative RMSE-Verbesserung
- Loss-Modi: `median_daily_rmse` (default) oder `pooled_rmse`
- Outputs:
  - `13_orientation_result.json`
  - `14_fit_diagnostics.csv`

#### kWp_effective

- Nach Orientierungswahl wird `kWp_effective` geschätzt:
  - `median(ac_power_clean / (POA_cs / 1000))`
  - nur für `fit_mask` und `POA_cs > 600 W/m²`
- Felder werden in `13_orientation_result.json` gespeichert.

### Block E (voll implementiert)

- Residual: `r = ac_power_clean / p_hat_unshaded_scaled`
- Filter für Residuale: fit-times, `POA_cs > 200`, `r > 0`
- 2D-Binning (Solar-Azimuth/Solar-Elevation) -> `shading_map.parquet`
- Indizes:
  - `global_shading_index`
  - `morning_shading_index` (Sektor default 60..150°)
  - `evening_shading_index` (Sektor default 210..300°)
- Plot:
  - `shading_map.png`
- Metriken:
  - `shading_metrics.json`

## Output-Layout

```text
outputs/<system_id>/<YYYYMMDD_HHMMSS>/
  01_parsed_tzaware.parquet
  02_cleaned_timeshift_fixed.parquet
  03_fit_times_mask.parquet
  04_daily_flags.csv
  05_clipping_summary.json
  06_clipped_times_mask.parquet
  07_sdt_summary.json
  07_sdt_introspect.json
  08_daily_peak.csv
  09_p_norm.parquet
  11_fit_mask.parquet
  12_daily_fit_fraction.csv
  13_orientation_result.json
  14_fit_diagnostics.csv
  shading_map.parquet
  shading_metrics.json
  shading_map.png
  summary.json
```

Aggregierter Report:

```text
outputs/reports/report_summary.csv
outputs/reports/report_summary.json
```

## CLI

```bash
pv-ident run-single --input-file <path> --metadata-json data/sonnja_pv3_2015/metadata.json --config examples/config.yml --output-dir outputs --system-id sonnja
pv-ident run-wide --config examples/config.yml --input examples/example_wide.csv
pv-ident run --config examples/config.yml --manifest examples/manifest.csv
pv-ident report --config examples/config.yml --output-root outputs
```
