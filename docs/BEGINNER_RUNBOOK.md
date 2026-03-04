# Beginner Runbook: pv-profiler Schritt für Schritt

Dieses Dokument ist für Nutzer ohne Programmierhintergrund.

## 1) Was macht das Tool?

`pv-profiler` schätzt aus deiner AC-Leistungskurve die Modul-Ausrichtung:
- Tilt (Neigung)
- Azimuth (Ausrichtung)

## 2) Welche Eingaben brauchst du mindestens?

- CSV-Datei mit:
  - Zeitspalte (z. B. `timestamp`)
  - Leistungsspalte (z. B. `P_AC`)
- Koordinaten:
  - Latitude
  - Longitude

Optional:
- `metadata.json` für `run-single`

## 3) Schnellster Weg: Alles in einem Lauf

Für einen vollständigen Lauf (Blocks 1–5) auf **einer** Datei:

```powershell
pv-ident run-batch `
  --input-dir "C:\Users\...\data\sonnja_pv3_2015" `
  --pattern "einleuchtend_wrdata_2015_wr1_5min_naive.csv" `
  --output-root "C:\Users\...\outputs\sonnja_wr1_all_blocks" `
  --timestamp-col "timestamp" `
  --power-col "P_AC" `
  --timezone "Etc/GMT-1" `
  --latitude 52.45544 `
  --longitude 13.52481 `
  --jobs 1
```

Warum `run-batch` auch für eine Datei?
- Es führt alle Blöcke robust durch.
- Schreibt Statusdateien.
- Nutzt klaren fixed-offset Parameter.

## 4) Was liegt danach im Output?

Im Zielordner entsteht pro System ein Unterordner mit Artefakten wie:
- `01_input_power.parquet`
- `02_sdt_daily_flags.csv`
- `05_power_fit.parquet`
- `07_p_norm_clear.parquet`
- `08_orientation_result.json`
- `00_status.json` (Status/Fehlerinfo)

Zusätzlich entsteht:
- `batch_summary.csv`

## 5) Wie erkennst du Erfolg?

- `00_status.json` enthält `"status": "ok"`.
- `08_orientation_result.json` existiert.
- `batch_summary.csv` zeigt Status pro System.

## 6) Wenn etwas fehlschlägt

- Öffne `00_status.json` für die Kurzursache.
- Öffne `00_error.txt` für Details (Traceback).
- Typische Ursachen:
  - Spaltenname falsch (`timestamp_col`, `power_col`)
  - Zu wenige Samples
  - Keine passenden Koordinaten

## 7) Einzelfall statt Batch

`run-single` ist bequem, wenn du nur eine Datei + metadata.json hast.
Hinweis: Für harte fixed-offset Kontrolle ist `run-batch` derzeit transparenter, da `--timezone` direkt gesetzt wird.

## 8) KI-freundliche Zusammenfassung (für Agenten/Automatisierung)

- Primärer End-to-End-Befehl: `pv-ident run-batch`
- Muss gesetzt werden:
  - `--input-dir`
  - `--pattern`
  - `--output-root`
  - `--timestamp-col`
  - `--power-col`
  - `--timezone` (empfohlen fixed offset: `Etc/GMT-1`)
  - `--latitude`, `--longitude` (oder Metadaten-Mapping für viele Systeme)
- Ergebnisprüfung:
  - global: `batch_summary.csv`
  - pro System: `00_status.json`, `08_orientation_result.json`
