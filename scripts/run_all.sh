#!/usr/bin/env bash
set -euo pipefail
python -m mm_tte_survival.cli make-demo-data --out data/demo --n 260 --p 120 --seed 42
python -m mm_tte_survival.cli audit-data --clinical data/demo/clinical.csv --cytogenetics data/demo/cytogenetics.csv --omics data/demo/omics.csv --out outputs/demo_audit.json
python -m mm_tte_survival.cli run-experiments --config configs/default.yaml
