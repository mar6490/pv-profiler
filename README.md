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
