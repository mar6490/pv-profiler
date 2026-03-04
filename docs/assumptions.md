# pv-profiler – Annahmen & Constraints

> Ziel: explizite Sammlung aller verwendeten Annahmen.  
> Markierung: **[REVIDIERT]** wenn im Verlauf geändert.

## A. Zeitachse / Timezone

- Loggerdaten sind „lokale Uhrzeit die einfach durchläuft“ (keine DST-Sprünge).
- Standard-Zeitzone als Fixed Offset:
  - `Etc/GMT-1` (Default/Fallback).
- `Europe/Berlin` wurde als riskant bewertet (DST).
- Es muss eine klare Prioritätsregel geben:
  1) tz-aware Index
  2) CLI `--timezone`
  3) Fallback Fixed Offset  
  **[UNSICHER]**: genaue Reihenfolge wurde diskutiert, aber nicht final als Spezifikation festgeschrieben.

## B. Daten / Sampling

- Sampling typ. 5‑min, regelmäßig (Pipeline erwartet Regularität).
- AC-Leistung ist primärer Signalträger für Orientierung.
- Clipping-Zeiten müssen ausgeschlossen werden.

## C. Clear/fit Zeiten

- SDT liefert:
  - daily clear flags
  - clipping flags
- `clear_times` (BooleanMasks) ist nicht zuverlässig verfügbar → muss robust ersetzt werden.
- Mindestkriterium: `clear_time_fraction_overall >= 0.02` sonst Fit instabil.

## D. Orientation Modell

- Forward-Modell via pvlib clearsky ist hinreichend für Orientierungsschätzung.
- Grid Search bevorzugt (Transparenz, Stabilität).

## E. Two-plane Systeme

- Früh: „zwei Dachflächen stehen ungefähr rechtwinklig“ → **falsch**.
- Später: „falls zwei Dachflächen: ungefähr 180°“ (Ost-West) **[REVIDIERT]**.
- Two-plane kann zusätzlich Mischgewicht (east/west) haben.

## F. Performance

- Full grid (insb. two-plane + weight grid) ist zu langsam → Optimierung notwendig.
- MVP: Fit gegen median minute_of_day Profil ist akzeptabel für erste stabile Version.
