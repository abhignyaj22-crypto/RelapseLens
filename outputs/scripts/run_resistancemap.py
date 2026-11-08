#!/usr/bin/env python3
"""Run the PROPOSED ResistanceMap × MMTTE pipeline end-to-end on real public data.

This is the dashboard proposal (https://abhignya-jagathpally.github.io/dashboard/#/science)
made concrete under the two-domain honesty constraint. It is DISTINCT from the repo's
canonical `make all` (the legacy single-modality survival model). Stages:

  0  fetch      public modalities (CCLE, STRING, DepMap, GEO) + CCLE RNA    [idempotent]
  D1 L1 VAE     CCLE proteome -> epigenome                                  ccle_paired + vae
  D2 PPI graph  STRING v12 over patient gene space
  D3 ESM-2      protein-sequence node features
  D4 L3 GNN     PPI-GNN vs degree-null (CRISPR essentiality)                exit gate
  E1 transfer   RNA->latent bridge -> patient transferred latent + OS ΔC
  E2 fusion     ResistanceModel(CrossAttentionFusion, CoxHead) OS ΔC        the proposed model
  F1 drug       bortezomib response (GSE9782), foundation priors

Each step is skipped if its artifact exists (pass --force to rerun). Ends by aggregating
every stage's JSON into one pipeline report.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DERIVED = ROOT / "data/external/_derived"
OUT = ROOT / "outputs"
FORCE = "--force" in sys.argv[1:]

# (label, script, artifact-that-means-done)
STEPS = [
    ("D1 fetch+prep CCLE paired", "scripts/stage_d/prep_ccle_multiomic.py", DERIVED / "ccle_paired.npz"),
    ("D1 L1 CellLineVAE", "scripts/stage_d/train_ccle_vae.py", OUT / "stage_d/ccle_vae_recon.json"),
    ("D2 STRING PPI graph", "scripts/stage_d/build_string_graph.py", DERIVED / "string_graph.npz"),
    ("D3 ESM-2 node features", "scripts/stage_d/esm2_node_features.py", DERIVED / "esm2_node_features.npz"),
    ("D4 L3 PPI-GNN null gate", "scripts/stage_d/train_ppi_gnn.py", OUT / "stage_d/ppi_gnn_null_gate.json"),
    ("E1 transfer + OS ΔC", "scripts/stage_e/transfer_and_evaluate.py", OUT / "stage_e/transfer_value.json"),
    ("E2 fusion model (proposed)", "scripts/stage_e/train_fusion_model.py", OUT / "stage_e/fusion_model_value.json"),
    ("F1 drug response GSE9782", "scripts/stage_f/drug_response_gse9782.py", OUT / "stage_f/drug_response_gse9782.json"),
]


def run(label, script, artifact):
    if artifact.exists() and not FORCE:
        print(f"[skip] {label} (artifact exists: {artifact.relative_to(ROOT)})", flush=True)
        return True
    print(f"\n===== {label}  ->  {script} =====", flush=True)
    r = subprocess.run([PY, str(ROOT / script)], cwd=ROOT)
    ok = r.returncode == 0
    print(f"[{'ok' if ok else 'FAIL'}] {label} (exit {r.returncode})", flush=True)
    return ok


def main():
    print("ResistanceMap × MMTTE — proposed pipeline, real public data only\n" + "=" * 64)
    results = {}
    for label, script, artifact in STEPS:
        if not run(label, script, artifact):
            print(f"\n[abort] {label} failed; fix and re-run (idempotent).")
            return 1

    # aggregate every stage JSON into one report
    report = {}
    for name, path in [
        ("L1_vae_reconstruction", OUT / "stage_d/ccle_vae_recon.json"),
        ("L3_ppi_gnn_null_gate", OUT / "stage_d/ppi_gnn_null_gate.json"),
        ("E_transfer_OS_value", OUT / "stage_e/transfer_value.json"),
        ("E_fusion_model_OS_value", OUT / "stage_e/fusion_model_value.json"),
        ("F_drug_response_gse9782", OUT / "stage_f/drug_response_gse9782.json"),
    ]:
        if path.exists():
            report[name] = json.loads(path.read_text())
    (OUT / "resistancemap_pipeline_report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 64 + "\nPIPELINE COMPLETE — headline numbers")
    g = report.get("L3_ppi_gnn_null_gate", {})
    print(f"  L1 VAE proteome->epigenome  median Spearman = {report.get('L1_vae_reconstruction',{}).get('median_spearman')}")
    print(f"  L3 PPI-GNN real vs null     = {g.get('real_graph',{}).get('spearman')} vs {g.get('degree_null',{}).get('spearman')} (gate {g.get('gate_real_beats_null')})")
    e = report.get("E_transfer_OS_value", {}); f = report.get("E_fusion_model_OS_value", {})
    print(f"  E transfer OS ΔC vs clinical = {e.get('dC_transfer_vs_clinical_mean_ci',[None])[0]} CI {e.get('dC_transfer_vs_clinical_mean_ci',[None,None,None])[1:]}")
    print(f"  E fusion   OS ΔC vs clinical = {f.get('dC_fusion_vs_clinical_mean')} CI {f.get('dC_ci95')} (gate {f.get('gate_fusion_beats_clinical')})")
    d = report.get("F_drug_response_gse9782", {})
    print(f"  F bortezomib AUROC  full={d.get('auroc_full_transcriptome')} proteasome={d.get('auroc_proteasome_signature')} crispr={d.get('auroc_crispr_essential_top500')}")
    print(f"\n  full report -> outputs/resistancemap_pipeline_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
