# Timezone Guide (Wichtig für korrekte Ergebnisse)

Diese Seite erklärt die Zeitzonen-Logik so, dass auch Nicht-Programmierer sicher arbeiten können.

## Kurzfassung (Empfehlung)

- Verwende **immer Fixed Offset** (z. B. `Etc/GMT-1`) für dieses Tool.
- **Nicht** `Europe/Berlin` verwenden, wenn du strikt keine Sommerzeit-/Winterzeit-Umschaltung willst.
- Für deine Sonnja-Daten ist in der Praxis meist sinnvoll: `--timezone Etc/GMT-1`.

## Warum ist das wichtig?

PV-Zeitreihen sind stark zeitabhängig. Schon eine Verschiebung um 1 Stunde kann Tilt/Azimuth verfälschen.

## DST vs. Fixed Offset

- `Europe/Berlin` = regionale Zeitzone mit DST (Sommer/Winter-Zeitwechsel).
- `Etc/GMT-1` = fester Offset ohne Wechsel (immer +01:00 Lokalzeit).

Wenn ein Datensatz konstant in „Logger-Lokalzeit“ aufgezeichnet wurde, ist ein fixer Offset oft robuster und reproduzierbarer.

## Wie das Tool intern mit Zeit umgeht

Block 1 (`load_input_for_sdt`) verarbeitet Timestamps so:

1. CSV-Zeitspalte wird geparst.
2. Wenn `--timezone` gesetzt ist:
   - bei tz-naiven Timestamps: `tz_localize(...)`
   - bei tz-aware Timestamps: `tz_convert(...)`
3. Danach wird für SDT mit naiver lokaler Zeit weitergearbeitet.

Das bedeutet: Das Tool versucht, doppelte Lokalisierung zu vermeiden, aber **dein gewählter Timezone-String entscheidet weiterhin maßgeblich über die Interpretation**.

## Entscheidungsbaum für Anwender

1. **Du willst garantiert kein DST-Verhalten**
   - Nutze `Etc/GMT-1` (oder anderen festen Offset).
2. **Deine Daten enthalten schon Offset (`+01:00`)**
   - Nutze weiterhin `Etc/GMT-1`, dann bleibt die Interpretation konsistent.
3. **Du bist unsicher**
   - Starte mit `Etc/GMT-1` und prüfe Plausibilität (Mittagsmaximum, Sonnenauf-/untergangsnahes Verhalten).

## Beispiel (Windows PowerShell, fixed offset)

```powershell
pv-ident run-batch `
  --input-dir "C:\path\to\data" `
  --pattern "ein_system.csv" `
  --output-root "C:\path\to\out" `
  --timestamp-col "timestamp" `
  --power-col "P_AC" `
  --timezone "Etc/GMT-1" `
  --latitude 52.45544 `
  --longitude 13.52481 `
  --jobs 1
```

## Häufige Fehlerbilder

- **Ergebnisse wirken um ~1 Stunde verschoben**
  - Prüfe ob versehentlich `Europe/Berlin` statt fixed offset genutzt wurde.
- **Uneinheitliche Jahresverläufe rund um März/Oktober**
  - Oft DST-Effekt; fixed offset verwenden.
