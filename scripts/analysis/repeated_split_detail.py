#!/usr/bin/env python
"""Analysis: repeated stratified-split stability (compute-only -> CSV).

Refits the penalised Cox ablations across N stratified splits and writes the
per-split values that SFig 2 plots. This is the *analysis* half of the
SFig 2 figure: it is the ONLY place those numbers are computed. The figure script
(scripts/figures/sfig2_repeated_split_stability.py) reads these CSVs and does no fitting.

Splitting compute (here) from plotting (there) keeps model-fitting on a single,
auditable path and stops a figure script from silently becoming the source of a
manuscript number (Fragility 1). See docs/figures/figure_manifest.md (S4 row).

Outputs (outputs/experiment0_open_gdc_os/):
  repeated_split_per_model.csv   long: split, feature_set, test_cindex, n_test, events_test
  repeated_split_paired_delta.csv long: split, comparison, delta_cindex
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mm_tte_survival.metrics import fast_c_index as cidx
from mm_tte_survival.data.cohort import build_matched_cohort
from mm_tte_survival.data.splits import stratified_event_split
from mm_tte_survival.evaluation.evaluate import _fit_cox

OUT = ROOT / "outputs" / "experiment0_open_gdc_os"
CFG = {"paths": {"clinical": str(ROOT / "data/real/clinical_survival.csv"),
                 "cytogenetics": str(ROOT / "data/real/cytogenetics.csv"),
                 "omics": str(ROOT / "data/real/omics.csv")},
       "schema": {"id_col": "patient_id", "time_col": "time_months", "event_col": "event",
                  "clinical_cols": ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy", "albumin", "b2m"],
                  "cytogenetic_cols": ["amp1q", "del1p", "del13q", "del17p", "t_4_14", "t_11_14", "t_14_16", "hyperdiploid"]}}
N_SPLITS, SEED, HORIZON = 50, 42, 24.0


def main():
    df, g = build_matched_cohort(CFG)
    tc, ec = "time_months", "event"
    sets = {"clinical": g["clinical"], "clinical+cyto": g["clinical"] + g["cyto"],
            "clinical+omics": g["clinical"] + g["omics"],
            "clinical+cyto+omics": g["clinical"] + g["cyto"] + g["omics"]}

    rows, deltas = [], []
    for s in range(N_SPLITS):
        tm = stratified_event_split(df, ec, SEED + s)
        te = ~tm
        t = pd.to_numeric(df[tc]).values[te.values]
        e = df[ec].astype(int).values[te.values]
        n_test, n_events = int(te.sum()), int(e.sum())
        r = {}
        for name, cols in sets.items():
            risk, _ = _fit_cox(df, cols, tm, tc, ec, HORIZON)
            r[name] = cidx(t, e, risk)
            rows.append({"split": s, "feature_set": name, "test_cindex": round(r[name], 6),
                         "n_test": n_test, "events_test": n_events})
        deltas.append({"split": s, "comparison": "+omics vs clinical",
                       "delta_cindex": round(r["clinical+omics"] - r["clinical"], 6)})
        deltas.append({"split": s, "comparison": "+cyto+omics vs clinical",
                       "delta_cindex": round(r["clinical+cyto+omics"] - r["clinical"], 6)})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "repeated_split_per_model.csv", index=False)
    pd.DataFrame(deltas).to_csv(OUT / "repeated_split_paired_delta.csv", index=False)
    print(f"wrote {OUT}/repeated_split_per_model.csv ({len(rows)} rows)")
    print(f"wrote {OUT}/repeated_split_paired_delta.csv ({len(deltas)} rows)")


if __name__ == "__main__":
    main()
