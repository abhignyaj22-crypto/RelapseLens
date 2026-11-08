"""mmSYGNAL external benchmark wrapper (research benchmarking only).

mmSYGNAL is a GPL-3.0 third-party RELAPSE-risk model (caret/glmnet) consuming
141-program activity (labels 0..140). This module:
  - shells out to scripts/benchmarks/run_mmsygnal.R to score patients,
  - merges scores onto the matched cohort (same patients only),
  - builds an endpoint-matched OS comparison table + a claim card.

HONESTY GUARDS:
  * mmSYGNAL needs mmSYGNAL/MINER program activity (0..140); RNA PCs and the
    repo's 10 curated `program_activity.csv` signatures are NOT valid inputs.
    If a real 141-program file is absent, this writes an honest "blocked" stub
    rather than fabricating scores.
  * mmSYGNAL targets relapse/PFS; scoring it on OS is a research benchmark of
    discrimination only, NOT a test of its intended PFS performance, and NOT a
    relapse/PFS claim for either model.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

PROGRAM_COLS = [str(i) for i in range(141)]


def run_mmsygnal_scoring(program_activity_csv, cytogenetics_csv, output_csv,
                         script_path="scripts/benchmarks/run_mmsygnal.R",
                         rscript="Rscript") -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [rscript, str(script_path), str(program_activity_csv),
           str(cytogenetics_csv), str(output_csv)]
    subprocess.run(cmd, check=True)
    return pd.read_csv(output_csv)


def merge_mmsygnal_scores(cohort: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" not in cohort.columns:
        raise ValueError("cohort must contain patient_id")
    if "patient_id" not in scores.columns:
        raise ValueError("mmSYGNAL scores must contain patient_id")
    cohort = cohort.copy(); scores = scores.copy()
    cohort["patient_id"] = cohort["patient_id"].astype(str)
    scores["patient_id"] = scores["patient_id"].astype(str)
    keep = ["patient_id"] + [c for c in scores.columns if c.startswith("mmsygnal_")]
    return cohort.merge(scores[keep], on="patient_id", how="inner")


def _has_real_program_activity(path: Path) -> bool:
    """True only if the file is a valid mmSYGNAL 141-program activity matrix.

    Delegates to the strict schema validator so RNA PCs and the 10-signature
    program_activity.csv are rejected (they lack columns 0..140)."""
    from .mmsygnal_schema import is_valid_mmsygnal_program_activity
    return is_valid_mmsygnal_program_activity(path)


def run_mmsygnal_benchmark(cfg: dict, bench_cfg: dict) -> dict:
    """Step 9: endpoint-matched OS comparison incl. mmSYGNAL (pretrained, no refit)."""
    from ..data.cohort import build_matched_cohort
    from ..data.splits import stratified_event_split
    from ..metrics import fast_c_index as cidx
    from lifelines import CoxPHFitter

    outdir = Path(cfg["paths"]["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    schema = cfg["schema"]; time_col, event_col = schema["time_col"], schema["event_col"]
    seed = int(cfg.get("seed", 42))
    prog_path = Path(bench_cfg["paths"]["program_activity"])
    scores_csv = Path(bench_cfg["benchmark"]["output"])

    _write_benchmark_claim_card(outdir, bench_cfg)

    if not _has_real_program_activity(prog_path):
        msg = (f"BLOCKED: {prog_path} does not contain mmSYGNAL 141-program activity "
               "(columns 0..140). mmSYGNAL scoring requires program activity generated "
               "by the mmSYGNAL/MINER inference pipeline (Wall et al. 2021) on the RNA. "
               "RNA PCs and the 10 curated program signatures are NOT valid inputs; no "
               "scores were fabricated.")
        pd.DataFrame([{"status": "blocked", "reason": msg}]).to_csv(
            outdir / "mmsygnal_comparison.csv", index=False)
        print(msg)
        return {"status": "blocked", "reason": msg}

    # ---- score with the upstream R models (pretrained, no refit) ----
    scores = run_mmsygnal_scoring(prog_path, bench_cfg["paths"]["cytogenetics"], scores_csv,
                                  rscript=bench_cfg.get("rscript", "Rscript"))
    df, groups = build_matched_cohort(cfg)
    df = merge_mmsygnal_scores(df, scores)            # same patients only
    tm = stratified_event_split(df, event_col, seed)
    te = ~tm
    t = pd.to_numeric(df[time_col]).values[te.values]
    e = df[event_col].astype(int).values[te.values]

    def cox_cindex(cols):
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.loc[tm.values].median()).fillna(0.0)
        X = (X - X.loc[tm.values].mean()) / X.loc[tm.values].std().replace(0, 1)
        d = X.copy(); d[time_col] = pd.to_numeric(df[time_col]).clip(lower=0.1).values
        d[event_col] = df[event_col].astype(int).values
        cph = CoxPHFitter(penalizer=0.1).fit(d[tm.values], duration_col=time_col, event_col=event_col)
        return cidx(t, e, cph.predict_log_partial_hazard(d[~tm.values][cols]).values)

    clin, cyto, om = groups["clinical"], groups["cyto"], groups["omics"]
    rows = [
        {"model": "clinical", "type": "fitted", "test_cindex": round(cox_cindex(clin), 4)},
        {"model": "clinical+cytogenetics", "type": "fitted", "test_cindex": round(cox_cindex(clin + cyto), 4)},
        {"model": "clinical+omics", "type": "fitted", "test_cindex": round(cox_cindex(clin + om), 4)},
        {"model": "clinical+cytogenetics+omics", "type": "fitted", "test_cindex": round(cox_cindex(clin + cyto + om), 4)},
        # mmSYGNAL pretrained scores used directly as risk (NO refit)
        {"model": "mmSYGNAL agnostic", "type": "pretrained_no_refit",
         "test_cindex": round(cidx(t, e, df["mmsygnal_agnostic_score"].values[te.values]), 4)},
        {"model": "mmSYGNAL selected subtype", "type": "pretrained_no_refit",
         "test_cindex": round(cidx(t, e, df["mmsygnal_selected_score"].values[te.values]), 4)},
        {"model": "clinical + mmSYGNAL selected", "type": "fitted",
         "test_cindex": round(cox_cindex(clin + ["mmsygnal_selected_score"]), 4)},
        {"model": "clinical+cytogenetics + mmSYGNAL selected", "type": "fitted",
         "test_cindex": round(cox_cindex(clin + cyto + ["mmsygnal_selected_score"]), 4)},
    ]
    for r in rows:
        r.update({"n_test": int(te.sum()), "events_test": int(e.sum()),
                  "endpoint": "overall_survival", "note": "OS research benchmark only; "
                  "mmSYGNAL targets relapse/PFS — NOT an endpoint-matched test of mmSYGNAL."})
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "mmsygnal_comparison.csv", index=False)
    out.to_csv(outdir / "sota_comparison.csv", index=False)
    print(out.to_string(index=False))
    return {"status": "ok", "comparison": out, "outdir": str(outdir)}


def _write_benchmark_claim_card(outdir: Path, bench_cfg: dict):
    L = ["# Benchmark claim card — mmSYGNAL vs proposed model", "",
         "- Source: github.com/baliga-lab/mmSYGNAL-risk-prediction-models (GPL-3.0), "
         "external dependency, not vendored.",
         "- Endpoint: **open_gdc_os (overall survival)**.",
         "- **mmSYGNAL is a relapse/PFS-risk model.** Evaluating it on OS measures OS "
         "discrimination only; it is NOT an endpoint-matched test of mmSYGNAL's intended "
         "PFS performance and gives mmSYGNAL no home-field advantage.",
         "- Rules enforced: same patients, same OS endpoint, same train/test split for "
         "fitted comparators; mmSYGNAL pretrained scores used with **no refitting**.",
         "- **direct relapse/PFS claim allowed: NO** · clinical-use claim: NO · "
         "research benchmark: YES.",
         "- Do NOT rank OS C-index against PFS/risk metrics without this warning."]
    (outdir / "benchmark_claim_card.md").write_text("\n".join(L) + "\n")
