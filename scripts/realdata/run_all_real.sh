#!/usr/bin/env bash
# Reproducible real-data pipeline: GDC MMRF-CoMMpass open data -> TTE benchmark.
# Run from the repo root (mm_tte_opisd/). Uses `python` on PATH; override with PY=...
# e.g. PY=/path/to/venv/bin/python bash scripts/realdata/run_all_real.sh
set -euo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/../.."

echo "== Step 1: clinical_survival.csv (GDC clinical API) =="
$PY scripts/realdata/fetch_clinical.py

echo "== Step 2a: download CNV Copy Number Segment files =="
$PY scripts/realdata/fetch_cnv.py
echo "== Step 2b: call cytogenetics from CNV =="
$PY scripts/realdata/call_cytogenetics.py

echo "== Step 3a: download STAR gene-counts (RNA) =="
$PY scripts/realdata/fetch_rna.py
echo "== Step 3b: build omics PCs + expression-surrogate translocations + provenance =="
$PY scripts/realdata/build_omics.py
echo "== Step 3c: build curated MM program-activity features (optional 2nd omics path) =="
$PY scripts/realdata/build_programs.py

echo "== Step 4: data audit =="
$PY -m mm_tte_survival.cli audit-data \
  --clinical data/real/clinical_survival.csv \
  --cytogenetics data/real/cytogenetics.csv \
  --omics data/real/omics.csv \
  --out outputs/real_audit.json

echo "== Step 5: train experiments (neural Cox/AFT/FHT/OPSD) =="
$PY -m mm_tte_survival.cli run-experiments --config configs/real_training.yaml

echo "== Step 6: benchmark table (classical Cox/AFT + subtype + neural) =="
$PY scripts/realdata/benchmark.py

echo "== Step 7: interpretation (importance, KM, calibration) =="
$PY scripts/realdata/interpret.py

echo "== Step 8: residual-risk decomposition + matched ablation + claim/usefulness =="
$PY -m mm_tte_survival.cli residual-report --config configs/real_training.yaml

echo "Done. See outputs/real_run/ and outputs/real_audit.{json,md}"
