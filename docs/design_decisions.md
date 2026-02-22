# pv-profiler – Design Decisions (für Projektbericht)

> Struktur: Entscheidung → Alternativen → Gründe → Konsequenzen → Status  
> Markierung: **[REVIDIERT]** wenn später geändert.

## DD-01: Forward-Modell (pvlib) statt Blackbox/ML
- Entscheidung: Orientation-Schätzung über physikalisches Modell.
- Alternativen:
  - ML/Blackbox
  - pv-analytics heuristics
  - SDT-internes Orientation-Modul
- Gründe:
  - Interpretierbarkeit / wissenschaftliche Begründbarkeit
  - Reproduzierbarkeit
  - Kontrollierbare Annahmen
- Konsequenzen:
  - Erfordert saubere Timezone/clear-time Selektion
  - Sensitiv auf Datenqualität (clear fraction)

## DD-02: SDT nur für Flags, nicht als Orientation-Kern
- Entscheidung:
  - SDT für clear/clipping/day flags.
  - Orientation-Fit eigenständig.
- Alternativen:
  - SDT `estimate_orientation` (oder ähnliche)
- Gründe:
  - In DACH zu wenig klare Tage → Instabilität/Fehlerberichte
  - API/Masken (clear_times) nicht zuverlässig dokumentiert

## DD-03: Fixed Offset Timezone (Etc/GMT-1) als Default
- Entscheidung:
  - DST vermeiden, Fixed Offset bevorzugen.
- Alternativen:
  - Europe/Berlin IANA (mit DST)
  - UTC-only
- Gründe:
  - Loggerdaten laufen ohne DST
  - SDT/Matrix-Indexing wird durch DST-Sprünge gestört
- Konsequenzen:
  - CLI muss Timezone-Policy transparent machen
  - Tests/Docs müssen DST-Fälle abdecken

## DD-04: Fit-Day/Time Auswahl als zentrales Stabilitätskriterium
- Entscheidung:
  - `is_fit_day = clear & not clipped & no_errors`
  - Exklusion bei `clear_time_fraction_overall < 0.02`
- Alternativen:
  - „Alles rein“ und robusten Fit nutzen
  - aggressives Glätten/Zacken-Filter
- Gründe:
  - Zu aggressive Filter entfernen Schatteninformation
  - Zu wenige clear times → Fit instabil

## DD-05: MVP Aggregation: median(p_norm) nach minute_of_day
- Entscheidung:
  - Fit gegen median-Profil (statt alle Samples).
- Alternativen:
  - Fit gegen alle Zeitpunkte
  - saisonale Subsets sofort
- Gründe:
  - robust gg. Ausreißer/Wolkenreste
  - deutliche Speedups
  - einfacher Debug (observed median profile vs model profile)
- Konsequenzen:
  - spätere Erweiterung muss vorgesehen werden (All-samples / seasonal)

## DD-06: Optimierung: Grid Search statt Nelder-Mead als Default
- Entscheidung:
  - Grid Search (transparent).
- Alternative:
  - Nelder-Mead / least-squares (continuous)
- Gründe:
  - Lokale Minima Risiko
  - Startwerte nötig
  - weniger auditierbar
- Konsequenzen:
  - Performancebedarf (insb. two-plane)

## DD-07: Two-plane Geometrie: ΔAz ≈ 180°
- Frühe Annahme:
  - ΔAz = 90° „fix“ (rechtwinklig) **[REVIDIERT]**
- Neue Spezifikation:
  - Ost-West: `az2 = (az1 + 180) % 360`
- Gründe:
  - fachliche Plausibilität für typische Ost-West Aufständerung / Dachflächen
- Konsequenzen:
  - Code muss Annahmen prüfen und dokumentieren
  - Tests/Checks für ΔAz implementieren

## DD-08: Two-plane Performance: weight-grid ersetzen
- Beobachtung:
  - Full grid + `weight_east` Grid ist Laufzeitkiller (~20 min in real runs)
- Entscheidung:
  - „massiv beschleunigen“; bevorzugt analytische Lösung für weight (lineare Mischung) oder Top‑k Refinement
- Alternativen:
  - grobes weight-grid beibehalten
- Gründe:
  - Laufzeit/Batchfähigkeit
- Status:
  - Umsetzung gefordert; genaue Implementierung im Chat angedeutet, aber nicht final spezifiziert. **[UNSICHER]**
