# pv-profiler

MVP-Pipeline zur Schätzung der Anlagenorientierung (Tilt/Azimuth) aus einer PV-AC-Leistungszeitreihe.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

pv-ident run-single \
  --input-csv data/sonnja_pv3_2015/einleuchtend_wrdata_2015_wr1_5min_naive.csv \
  --metadata data/sonnja_pv3_2015/metadata.json \
  --power-column P_AC
```

## Block 1 separat testen (Input Loader)

Wenn du **nur bis Block 1** testen willst (ohne vollständige Orientierungs-Schätzung), nutze:

```bash
pv-ident run-block1 \
  --input-csv data/sonnja_pv3_2015/einleuchtend_wrdata_2015_wr1_5min_naive.csv \
  --output-dir outputs/sonnja_wr1_block1 \
  --timestamp-col timestamp \
  --power-col P_AC
```

Erwartete Artefakte im Output-Ordner:
- `01_input_power.parquet`
- `01_input_diagnostics.json`

Kurzcheck:

```bash
python - <<'PY'
import json
import pandas as pd

df = pd.read_parquet("outputs/sonnja_wr1_block1/01_input_power.parquet")
diag = json.load(open("outputs/sonnja_wr1_block1/01_input_diagnostics.json", "r", encoding="utf-8"))

print(df.head())
print(df.columns.tolist())
print(diag["dominant_timedelta"], diag["shape"], diag["share_nan_power"])
PY
```

Optional kann das Ergebnis in eine JSON-Datei geschrieben werden:

```bash
pv-ident run-single \
  --input-csv data/sonnja_pv3_2015/einleuchtend_wrdata_2015_wr1_5min_naive.csv \
  --metadata data/sonnja_pv3_2015/metadata.json \
  --output-json data/processed/run_single_result.json
```

## Projektstruktur

- `src/pv_profiler/cli.py`: CLI (`pv-ident`)
- `src/pv_profiler/pipeline.py`: Orchestrierung
- `src/pv_profiler/block_io.py`: CSV/Metadata I/O
- `src/pv_profiler/block_sdt.py`: SDT Onboarding
- `src/pv_profiler/block_orientation.py`: Tilt/Azimuth-Schätzung mit pvlib
- `src/pv_profiler/block_diagnostics.py`: Qualitätskennzahlen
- `src/pv_profiler/types.py`: Dataclasses für Outputs
