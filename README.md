# pv-profiler

MVP-Pipeline zur Schätzung der PV-Anlagenorientierung (Tilt/Azimuth) aus einer AC-Leistungszeitreihe.

> Zielgruppe: Dieses README ist bewusst **anfängerfreundlich** geschrieben.

## Inhaltsverzeichnis

1. [Was das Tool macht](#was-das-tool-macht)
2. [Wichtiger Hinweis zur Zeitzone](#wichtiger-hinweis-zur-zeitzone)
3. [Installation](#installation)
4. [Schnellstart (ein System, alle Blöcke)](#schnellstart-ein-system-alle-blöcke)
5. [CLI-Befehle im Überblick](#cli-befehle-im-überblick)
6. [Eingabedaten richtig vorbereiten](#eingabedaten-richtig-vorbereiten)
7. [Output-Dateien verstehen](#output-dateien-verstehen)
8. [Fehlerbehebung (Troubleshooting)](#fehlerbehebung-troubleshooting)
9. [Batch- und Benchmark-Workflow](#batch--und-benchmark-workflow)
10. [Erweiterte Doku](#erweiterte-doku)

---

## Was das Tool macht

`pv-profiler` verarbeitet Leistungsdaten in mehreren Schritten („Blocks 1–5“):

- **Block 1:** CSV laden, Zeitreihe bereinigen/standardisieren.
- **Block 2:** SDT-Onboarding (Datenqualität/Flags).
- **Block 3:** Auswahl geeigneter Fit-Tage.
- **Block 4:** Normalisierung auf Tagesprofil.
- **Block 5:** Schätzung von Tilt/Azimuth (single oder two-plane Modell).

Für jeden Lauf entstehen nachvollziehbare Artefakte (CSV/JSON/Parquet).

---

## Wichtiger Hinweis zur Zeitzone

### Kurzregel

Wenn du **keine Sommer-/Winterzeit-Effekte (DST)** willst, nutze immer einen **fixed offset**:

- empfohlen: `Etc/GMT-1`

Nicht empfohlen für fixed-offset-Workflows:

- `Europe/Berlin` (hat DST)

### Warum?

Ein 1h-Versatz kann die Orientierungsschätzung sichtbar beeinflussen.

### Detaillierte Erklärung

Siehe: [`docs/TIMEZONE_GUIDE.md`](docs/TIMEZONE_GUIDE.md)

---

## Installation

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

---

## Schnellstart (ein System, alle Blöcke)

Für einen vollständigen End-to-End-Lauf über **eine CSV-Datei** nutze `run-batch` mit einem Dateimuster.

### Windows PowerShell (fixed offset)

```powershell
pv-ident run-batch `
  --input-dir "C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\data\sonnja_pv3_2015" `
  --pattern "einleuchtend_wrdata_2015_wr1_5min_naive.csv" `
  --output-root "C:\Users\Maren.Murjahn\PycharmProjects\pv-profiler\outputs\sonnja_wr1_all_blocks" `
  --timestamp-col "timestamp" `
  --power-col "P_AC" `
  --timezone "Etc/GMT-1" `
  --latitude 52.45544 `
  --longitude 13.52481 `
  --jobs 1
```

---

## CLI-Befehle im Überblick

### 1) `run-single`

Einfacher Ein-Datei-Workflow mit `metadata.json`.

```bash
pv-ident run-single \
  --input-csv data/sonnja_pv3_2015/einleuchtend_wrdata_2015_wr1_5min_naive.csv \
  --metadata data/sonnja_pv3_2015/metadata.json \
  --power-column P_AC
```

### 2) `run-batch`

Empfohlen für reproduzierbare Runs (auch bei nur einer Datei möglich), inklusive Statusdateien pro System.

```bash
pv-ident run-batch \
  --input-dir path/to/systems \
  --pattern "system_*.csv" \
  --output-root outputs/batch \
  --timestamp-col time \
  --power-col ac_power_w \
  --timezone Etc/GMT-1 \
  --jobs 4
```

### 3) Einzelne Blöcke separat

Nützlich zum Debugging.

#### Block 1

```bash
pv-ident run-block1 \
  --input-csv data/sonnja_pv3_2015/einleuchtend_wrdata_2015_wr1_5min_naive.csv \
  --output-dir outputs/sonnja_wr1_block1 \
  --timestamp-col timestamp \
  --power-col P_AC \
  --timezone Etc/GMT-1
```

#### Block 2

```bash
pv-ident run-block2 \
  --input-parquet outputs/sonnja_wr1_block1/01_input_power.parquet \
  --output-dir outputs/sonnja_wr1_block2 \
  --solver CLARABEL
```

#### Block 3

```bash
pv-ident run-block3 \
  --input-power-parquet outputs/sonnja_wr1_block1/01_input_power.parquet \
  --input-daily-flags-csv outputs/sonnja_wr1_block2/02_sdt_daily_flags.csv \
  --output-dir outputs/sonnja_wr1_block3
```

#### Block 4

```bash
pv-ident run-block4 \
  --input-power-fit-parquet outputs/sonnja_wr1_block3/05_power_fit.parquet \
  --output-dir outputs/sonnja_wr1_block4
```

#### Block 5

```bash
pv-ident run-block5 \
  --input-p-norm-parquet outputs/sonnja_wr1_block4/07_p_norm_clear.parquet \
  --output-dir outputs/sonnja_wr1_block5 \
  --latitude 52.45544 \
  --longitude 13.52481 \
  --timezone Etc/GMT-1
```

---

## Eingabedaten richtig vorbereiten

## Minimalanforderungen

- CSV mit mindestens:
  - Zeitspalte (`timestamp_col`)
  - Leistungsspalte (`power_col`, AC-Leistung)

## Typische Spalten aus deinem Datensatz

- Zeit: `timestamp`
- Leistung: `P_AC`
- Standort: `lat`, `lon` (optional im CSV, aber für Block 5 müssen Koordinaten über CLI oder Metadaten vorliegen)

## Zeitzonenformat in den Daten

- Timestamps können naiv oder tz-aware sein.
- Für reproduzierbaren Betrieb fixed offset verwenden (`Etc/GMT-1`).

---

## Output-Dateien verstehen

Typischerweise findest du im Output pro System:

- `01_input_power.parquet` – bereinigte Leistung
- `01_input_diagnostics.json` – Loader-Diagnose
- `02_sdt_daily_flags.csv` – SDT-Tagesflags
- `05_power_fit.parquet` – Fit-Zeitreihe
- `07_p_norm_clear.parquet` – normalisierte Kurve
- `08_orientation_result.json` – Kernergebnis (Tilt/Azimuth etc.)
- `00_status.json` – Status (ok/failed/skipped)
- `00_error.txt` – Stacktrace bei Fehlern

Zusätzlich im Batch-Root:

- `batch_summary.csv`

---

## Fehlerbehebung (Troubleshooting)

### Problem: Lauf schlägt sofort fehl

Prüfen:

- Stimmen `--timestamp-col` und `--power-col` wirklich exakt mit der CSV überein?
- Ist genug Datenmenge vorhanden (`min_samples`)?

### Problem: Ergebnis wirkt zeitlich verschoben

- Wurde versehentlich `Europe/Berlin` statt fixed offset genutzt?
- Für harte Reproduzierbarkeit `--timezone Etc/GMT-1` setzen.

### Problem: Batch liefert `failed` für einzelne Systeme

- In `00_status.json` und `00_error.txt` im jeweiligen Systemordner nachsehen.
- Batch läuft robust weiter; ein Systemfehler stoppt nicht den gesamten Lauf.

---

## Batch- und Benchmark-Workflow

```bash
pv-ident run-batch \
  --input-dir path/to/systems \
  --pattern "system_*.csv" \
  --output-root pvprofiler_benchmark \
  --timestamp-col time \
  --power-col ac_power_w \
  --latitude 52.45544 \
  --longitude 13.52481 \
  --timezone Etc/GMT-1 \
  --jobs 4

python scripts/benchmark_synthetic.py \
  --output-root pvprofiler_benchmark \
  --systems-metadata-csv path/to/systems_metadata.csv

python scripts/plot_benchmark_results.py \
  --input pvprofiler_benchmark/benchmark_results.csv
```

---

## Erweiterte Doku

- Zeitzonen ausführlich: [`docs/TIMEZONE_GUIDE.md`](docs/TIMEZONE_GUIDE.md)
- Anfänger-Schrittfolge: [`docs/BEGINNER_RUNBOOK.md`](docs/BEGINNER_RUNBOOK.md)

---

## Projektstruktur

- `src/pv_profiler/cli.py`: CLI (`pv-ident`)
- `src/pv_profiler/pipeline.py`: Orchestrierung
- `src/pv_profiler/block_io.py`: CSV/Metadata I/O
- `src/pv_profiler/block_sdt.py`: SDT Onboarding
- `src/pv_profiler/block_orientation.py`: Tilt/Azimuth-Schätzung mit pvlib
- `src/pv_profiler/block_diagnostics.py`: Qualitätskennzahlen
- `src/pv_profiler/types.py`: Dataclasses
