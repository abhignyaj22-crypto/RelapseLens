"""Smoke test for the refactored layered package + main.run_pipeline."""
from pathlib import Path
import pandas as pd
import yaml

from mm_tte_survival.demo import make_demo_data
from mm_tte_survival.main import run_pipeline
from mm_tte_survival.models.residual_risk import ResidualRiskModel
from mm_tte_survival.config import load_app_config
from mm_tte_survival.data.splits import stratified_event_split


def test_residual_risk_model_additive(tmp_path):
    data = tmp_path / "demo"
    make_demo_data(data, n=140, p=10, seed=3)
    df = pd.read_csv(data / "clinical.csv")
    df["patient_id"] = df["patient_id"].astype(str)
    cyto = pd.read_csv(data / "cytogenetics.csv"); cyto["patient_id"] = cyto["patient_id"].astype(str)
    df = df.merge(cyto, on="patient_id", how="left")
    clin = ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy"]
    mol = ["amp1q", "del1p", "del13q", "del17p"]
    tm = stratified_event_split(df, "event", 3)
    m = ResidualRiskModel(clin, mol).fit(df[tm.values])
    pred = m.predict(df)
    # total must equal clinical + molecular_residual exactly
    err = (pred.clinical_risk + pred.molecular_residual_risk - pred.total_risk).abs().max()
    assert err < 1e-9


def test_main_pipeline_demo(tmp_path):
    data = tmp_path / "demo"
    make_demo_data(data, n=170, p=12, seed=5)
    outdir = tmp_path / "out"
    cfg = {
        "seed": 5, "endpoint": {"name": "open_gdc_os"},
        "paths": {"clinical": str(data / "clinical.csv"), "cytogenetics": str(data / "cytogenetics.csv"),
                  "omics": str(data / "omics.csv"), "outdir": str(outdir)},
        "schema": {"id_col": "patient_id", "time_col": "time_months", "event_col": "event",
                   "clinical_cols": ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy"],
                   "cytogenetic_cols": ["amp1q", "del1p", "del13q", "del17p", "t_4_14", "t_11_14", "hyperdiploid"]},
        "cohort": {"matched_ablation": True},
        "evaluation": {"repeated_splits": 3, "horizon_months": 24, "min_test_events": 5,
                       "min_subtype_events_hypothesis": 5, "min_subtype_events_confirmatory": 15},
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    cfg_obj = load_app_config(str(tmp_path / "cfg.yaml"))
    assert cfg_obj.endpoint.name == "open_gdc_os"
    assert cfg_obj.to_dict()["validation"]["repeated_splits"] == 3
    results = run_pipeline(str(tmp_path / "cfg.yaml"))
    # all mandated artifacts + cards produced
    for f in ["matched_ablation.csv", "paired_delta_cindex.csv", "claim_report.json",
              "leakage_audit.json", "model_card.md", "claim_card.md", "data_card.md",
              "validation_report.json", "mmrf_reclassification_outcomes.csv"]:
        assert (outdir / f).exists(), f
    assert results["claim_report"]["technical_validation_claim_allowed"] is True
    assert results["claim_report"]["relapse_or_pfs_claim_allowed"] is False
