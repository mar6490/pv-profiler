# CODEX_CONTEXT.md — pv-profiler

Zweck: Stabiler Kontext für Codex/Code-Generatoren.  
Regeln: nicht „kreativ“ werden, sondern exakt die Spezifikation umsetzen.

## 1. Projektziel
Implementiere/erweitere `pv-profiler` zur Orientierungsschätzung (tilt/azimuth) aus AC‑Zeitreihen ohne Wetterdaten, inkl. Two-plane Ost-West und Performance-Optimierungen.

## 2. Nicht verhandelbare Constraints
- Sampling: 5‑min regular.
- Zeitzone: Default `Etc/GMT-1` (Fixed Offset), DST vermeiden.
- SDT: nur Flags (clipping, clear day/time), nicht als Orientation-Kern.
- Pipeline: Block A–E mit persistenten Artefakten.
- Outputs müssen deterministisch und reproduzierbar sein.

## 3. Timezone Policy (muss im Code explizit stehen)
Priorität:
1) tz-aware timestamps → respektieren
2) sonst CLI `--timezone`
3) sonst Fallback `Etc/GMT-1`
Logge die gewählte Policy und schreibe sie in Ergebnis-JSON.

## 4. Fit Mask Policy
- `is_fit_day = clear & not clipped & no_errors`
- Ausschluss wenn `clear_time_fraction_overall < 0.02`
- Filter dürfen Schatteninfo nicht löschen (kein aggressives „Zacken entfernen“ ohne Option).

## 5. Orientation Search
### Single-plane
- Grid Search tilt/azimuth
- Score: RMSE (optional BIC zusätzlich)

### Two-plane
- Default-Geometrie: `az2 = (az1 + 180) % 360`
- Keine 90° Annahme.
- Mischgewicht `weight_east` erlaubt, aber **nicht** per teurem weight-grid, wenn analytisch lösbar.

## 6. Performance
- MVP: median(p_norm) nach minute_of_day als Zielprofil.
- Implementiere Top‑k Refinement: erst grobes Grid, dann fein um Top‑k.
- Für two-plane: analytische Gewichtsschätzung prüfen/implementieren (lineare Mischung).

## 7. Artefakte (müssen erzeugt werden)
- `08_orientation_result.json`
- `09a_orientation_single_full_grid.csv`
- `09b_orientation_two_plane_full_grid.csv`
- `09_orientation_topk.csv`
- `10_profile_compare.csv`

## 8. Dokumentationspflicht
- Jede Annahme/Default muss in `docs/assumptions.md` + README (kurz) dokumentiert sein.
- Änderungen an Two-plane-Geometrie und Timezone müssen in `docs/design_decisions.md` nachvollziehbar sein.
