"""High-level production entrypoint — the single canonical execution path.

Canonical stages (Fragility 1: one auditable path to the manuscript):

    load -> validate -> preprocess -> fit -> evaluate -> report   [this module]
                                                          -> figure [make figures]

`run_pipeline` covers load..report. Preprocess+fit are train-only and live inside
`evaluate_model_suite` (compute-only; see docs/phase1_verification.md). The figure
stage is plot-only and orchestrated by the Makefile, so the full record regenerates
with a single command: `make all` (= run -> analysis -> figures).

The default production model is the interpretable ResidualRiskModel. The neural
MultiHeadSurvivalModel / OPSD heads are a parallel EXPERIMENTAL path (`make
exploratory`), not part of this canonical path and not the headline claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mm_tte_survival.config import load_app_config, set_seed
from mm_tte_survival.endpoints import resolve_endpoint
from mm_tte_survival.data.loaders import load_modalities
from mm_tte_survival.data.contracts import validate_all_inputs
from mm_tte_survival.evaluation.evaluate import evaluate_model_suite
from mm_tte_survival.reports.run_reports import write_all_reports


def run_pipeline(config_path: str) -> dict:
    cfg = load_app_config(config_path)
    set_seed(cfg.seed)
    raw_cfg = cfg.to_dict()

    # endpoint + claim context
    endpoint = resolve_endpoint(raw_cfg)

    # [load] data loading
    raw = load_modalities(
        clinical_path=cfg.paths.clinical,
        cytogenetics_path=cfg.paths.cytogenetics,
        omics_path=cfg.paths.omics,
        program_activity_path=getattr(cfg.paths, "program_activity", None),
    )

    # [validate] contracts / leakage / provenance checks
    validate_all_inputs(raw=raw, endpoint=endpoint, schema=raw_cfg["schema"],
                        output_dir=cfg.paths.outdir)

    # [preprocess + fit + evaluate] matched cohort, train-only preprocessing,
    # residual-risk model, endpoint-correct evaluation suite (compute-only)
    results = evaluate_model_suite(raw_cfg)

    # [report] endpoint-gated reports + cards (write-only). [figure] -> `make figures`
    outdir = write_all_reports(raw_cfg, results)

    c = results["claim_report"]
    print(f"[{endpoint['name']} / {endpoint['endpoint_type']}] "
          f"technical_validation={c['technical_validation_claim_allowed']} "
          f"relapse_pfs={c['relapse_or_pfs_claim_allowed']} "
          f"omics={c['omics_increment_summary']} -> {outdir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MM-TTE residual-risk pipeline")
    parser.add_argument("--config", default="configs/experiments/experiment0_open_gdc_os.yaml",
                        help="Path to experiment config.")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
