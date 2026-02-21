# pv-profiler – Architektur

> Ziel dieses Dokuments: Architektur 
> Kennzeichnung: **[UNSICHER]** = angedeutet, aber nicht eindeutig finalisiert / mehrfach widersprüchlich.

## 1. Systemziel und Scope

- Eingabe: **AC-Leistungszeitreihe** (typ. 5‑min Raster), optional zusätzlich DC (nur wenn vorhanden).
- Ausgabe:
  - Schätzung **Azimut** und **Neigung** (Single-Plane und optional Two-Plane / Ost-West).
  - Diagnose-/Vergleichsartefakte (Top‑k Kandidaten, Profilvergleich, Fit-Güte).
  - Flags/Masken für „fit times“/„clear times“/Clipping/Fehler.

- Primärannahme: **ohne externe Wetterdaten**, Orientierung aus AC-only über **Forward-Clearsky-Modell**.

## 2. High-level Architektur: Pipeline Blocks (A–E)

Die Implementierung ist als Pipeline mit persistenten Zwischenartefakten geplant/umgesetzt.

### Block A — Input, Zeitachse, Basis-Cleaning
**Ziele**
- Robustes Timestamp-Parsing (naive vs tz-aware).
- Einheitliche Zeitbasis (DST-freies Verhalten).
- Aufbau einer SDT-kompatiblen Datenmatrix inkl. Grundreinigung.

**Kernpunkte aus dem Chat**
- Loggerdaten laufen „lokal durch“ (keine DST-Umstellung); deshalb wird *Europe/Berlin* als riskant betrachtet.
- Default/Fallback Zeitzone: **Etc/GMT-1** (Fixed Offset).
- CLI-Override `--timezone` vorgesehen.

**Artefakte**
- `01_input_power.parquet` (Input)
- `02_cleaned_timeshift_fixed.parquet` (bereinigt + zeitlich konsistent)

**Unsicherheiten**
- Prioritätsregel tz-aware vs CLI vs Fallback wurde diskutiert; im Code muss sie explizit festgelegt werden. [UNSICHER]

### Block B — Masken, Tagesflags, Fit-Definition
**Ziele**
- Ermittlung von Tages-/Zeitpunkt-Flags:
  - clear day / clear times
  - inverter clipping
  - error/noise flags
- Ableitung einer **Fit-Maske** (welche Zeitpunkte/Tage dürfen in Block C/D genutzt werden).

**SDT-Integration**
- SDT wird primär für „daily flags“ und clipping/clear day Klassifikation genutzt.
- Kritischer Punkt: `dh.boolean_masks.clear_times` ist in realen Runs oft `None` und nicht zuverlässig dokumentiert → alternative Clear-Time-Logik erforderlich.

**Artefakte**
- `03_clear_times_mask.parquet` (boolean mask / fit times)
- `02_sdt_daily_flags.csv` (DailyFlags Export; Name kann variieren)
- `05_power_fit.parquet` (normalisierte/fit-relevante Zeitreihe; je nach Implementierung)

**Fit-day Definition (rekonstruiert)**
- `is_fit_day = clear == True AND inverter_clipped == False AND no_errors == True`

**Schwellen/Heuristiken**
- `clear_time_fraction_overall >= 0.02` als Minimalanforderung für Stabilität (sonst Ausschluss).

### Block C — Forward-Modell (pvlib)
**Ziele**
- Erzeuge Modellprofil(e) für Kandidaten (tilt/azimuth) via pvlib:
  - clearsky (Ineichen o.ä.)
  - Transposition auf POA
  - (vereinfachte) Umrechnung zu normalisierter Leistung

**MVP-Aggregation**
- Statt Fit gegen alle Zeitpunkte: Aggregation der beobachteten Zeitreihe:
  - `median(p_norm)` nach `minute_of_day` über Fit-Tage → „Zielprofil“.
- Vergleich/Score Modellprofil gegen dieses Medianprofil (schnell/robust).

**Artefakte**
- Zwischenprofile (observed vs model) für Plot/CSV/Debug (Dateinamen variieren).
- `10_profile_compare.csv` (Profilvergleich; auch Block E möglich)

### Block D — Orientation Optimization
**Ziele**
- Suche nach tilt/azimuth mit minimalem Fehlermaß (RMSE; BIC wurde diskutiert).

**Single-Plane**
- Full grid → Export:
  - `09a_orientation_single_full_grid.csv`

**Two-Plane (Ost-West)**
- Zwei Dächer/Flächen:
  - Ursprünglich (frühe Diskussion): `delta_az_deg = 90` (rechtwinklig).
  - Später revidiert: **≈180°** (typ. Ost/West).
- Zusätzlich: Mischgewicht `weight_east` (0..1) zur Mischung East/West Profile.
- Full grid Export:
  - `09b_orientation_two_plane_full_grid.csv`

**Ergebnis**
- `08_orientation_result.json` (beste Parameter, Scores, Metadaten)
- `09_orientation_topk.csv` (Top‑k Kandidaten)

**Performance-Fokus**
- Two-plane brute-force „weight grid“ ist Laufzeitkiller → soll analytisch oder über geschickte Reduktion beschleunigt werden (Details in `design_decisions.md`).

### Block E — Residual- & Shading-Analyse
**Ziele**
- Residuen zwischen beobachtetem Profil und best-fit Modell analysieren.
- Ableitung von Shading-Indikatoren (nicht vollständig finalisiert).
- Diskussion über pv-analytics `features.shading` als potenzielle Abkürzung, aber nicht als Kernpfad.

## 3. CLI / Repo-Architektur (rekonstruiert)

- CLI-Kommandos:
  - `pv-ident --help`
  - `pv-ident run -block5` bzw. `pv-ident run-block5` (mehrere Varianten im Chat; konsolidieren!)
- Tests:
  - `pytest`
- Installation:
  - `pip install -e .`
- Repo Struktur (mindestens):
  - `pyproject.toml`
  - `README.md`
  - `src/` (Modulname: `pv_ident` / `pv_profiler` wurde diskutiert)

## 4. Artefakt-Namensschema (explizit genannt)
- `01_input_power.parquet`
- `02_cleaned_timeshift_fixed.parquet`
- `03_clear_times_mask.parquet`
- `05_power_fit.parquet`
- `08_orientation_result.json`
- `09a_orientation_single_full_grid.csv`
- `09b_orientation_two_plane_full_grid.csv`
- `09_orientation_topk.csv`
- `10_profile_compare.csv`
