#!/usr/bin/env python
"""RNA PCA vs official miner3/mmSYGNAL program activity (OS, matched cohort).

Answers: are biologically-structured miner3 programs better than generic RNA PCs
in the residual-risk framework? Same cohort, same split, same repeated-split
validation, same OS endpoint. For fairness, both omics representations are capped
to the SAME feature count (16): top-16 RNA PCs vs top-16 most-variable miner3
programs.

Outputs (outputs/experiment0_open_gdc_os/):
  program_vs_pca_ablation.csv, program_vs_pca_paired_delta.csv,
  program_vs_pca_calibration.csv, program_vs_pca_claim_card.md
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mm_tte_survival.metrics import fast_c_index as cidx
from mm_tte_survival.data.splits import stratified_event_split
from mm_tte_survival.evaluation.stats import paired_delta_cindex, calibration_metrics

REAL = ROOT / "data" / "real"
OUT = ROOT / "outputs" / "experiment0_open_gdc_os"
SEED, HORIZON, NF = 42, 24.0, 16
CLIN = ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy", "albumin", "b2m"]
CYTO = ["amp1q", "del1p", "del13q", "del17p", "t_4_14", "t_11_14", "t_14_16", "hyperdiploid"]
TIME, EVENT = "time_months", "event"


def load():
    clin = pd.read_csv(REAL / "clinical_survival.csv"); clin["patient_id"] = clin.patient_id.astype(str)
    cyto = pd.read_csv(REAL / "cytogenetics.csv"); cyto["patient_id"] = cyto.patient_id.astype(str)
    pca = pd.read_csv(REAL / "omics.csv"); pca["patient_id"] = pca.patient_id.astype(str)
    prog = pd.read_csv(REAL / "mmsygnal_program_activity_0_140.csv"); prog["patient_id"] = prog.patient_id.astype(str)
    pca_cols = [c for c in pca.columns if c.startswith("PC")][:NF]
    prog_all = [c for c in prog.columns if c != "patient_id"]
    # top-NF most-variable programs (fair, same count as PCs)
    var = prog[prog_all].var().sort_values(ascending=False)
    prog_cols = [f"prog_{c}" for c in var.index[:NF]]
    prog = prog.rename(columns={c: f"prog_{c}" for c in prog_all})
    df = clin.merge(cyto[["patient_id"] + CYTO], on="patient_id", how="left") \
             .merge(pca[["patient_id"] + pca_cols], on="patient_id", how="left") \
             .merge(prog[["patient_id"] + prog_cols], on="patient_id", how="left")
    df = df[pd.to_numeric(df[TIME], errors="coerce").gt(0)]
    keep = df[pca_cols].notna().all(axis=1) & df[prog_cols].notna().all(axis=1) & df[CYTO].notna().any(axis=1)
    return df[keep].reset_index(drop=True), [c for c in CLIN if c in df.columns], pca_cols, prog_cols


def fit(df, cols, tm, horizon):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.loc[tm.values].median()).fillna(0.0)
    X = (X - X.loc[tm.values].mean()) / X.loc[tm.values].std().replace(0, 1)
    d = X.copy(); d[TIME] = pd.to_numeric(df[TIME]).clip(lower=0.1).values; d[EVENT] = df[EVENT].astype(int).values
    cph = CoxPHFitter(penalizer=0.1).fit(d[tm.values], duration_col=TIME, event_col=EVENT)
    te = d[~tm.values]
    risk = cph.predict_log_partial_hazard(te[cols]).values
    p = 1.0 - cph.predict_survival_function(te[cols], times=[horizon]).iloc[0].values
    return risk, p


def main():
    df, clin, pca_cols, prog_cols = load()
    tm = stratified_event_split(df, EVENT, SEED)
    te = ~tm
    t = pd.to_numeric(df[TIME]).values[te.values]; e = df[EVENT].astype(int).values[te.values]
    sets = {
        "clinical": clin,
        "clinical+RNA_PCA": clin + pca_cols,
        "clinical+miner3_programs": clin + prog_cols,
        "clinical+cyto+RNA_PCA": clin + CYTO + pca_cols,
        "clinical+cyto+miner3_programs": clin + CYTO + prog_cols,
        "clinical+cyto+RNA_PCA+miner3_programs": clin + CYTO + pca_cols + prog_cols,
    }
    risks, probs, rows, cal = {}, {}, [], []
    for name, cols in sets.items():
        risk, p = fit(df, cols, tm, HORIZON)
        risks[name], probs[name] = risk, p
        c = cidx(t, e, risk)
        rng = np.random.default_rng(SEED)
        boots = [cidx(t[b], e[b], risk[b]) for b in (rng.integers(0, len(t), len(t)) for _ in range(500))]
        boots = [x for x in boots if np.isfinite(x)]
        rows.append({"feature_set": name, "n_features": len(cols), "test_cindex": round(c, 4),
                     "ci_low": round(np.percentile(boots, 2.5), 4), "ci_high": round(np.percentile(boots, 97.5), 4),
                     "n_test": int(te.sum()), "events_test": int(e.sum())})
        m, _ = calibration_metrics(t, e, p, HORIZON)
        cal.append({"feature_set": name, **m})

    # paired deltas: programs vs PCA (same test patients)
    pairs = [("clinical+miner3_programs", "clinical+RNA_PCA"),
             ("clinical+cyto+miner3_programs", "clinical+cyto+RNA_PCA"),
             ("clinical+cyto+RNA_PCA+miner3_programs", "clinical+cyto+RNA_PCA"),
             ("clinical+cyto+RNA_PCA+miner3_programs", "clinical+cyto+miner3_programs")]
    deltas = [{"comparison": f"{a}_vs_{b}", **paired_delta_cindex(t, e, risks[a], risks[b], seed=SEED)}
              for a, b in pairs]

    OUT.mkdir(parents=True, exist_ok=True)
    abl = pd.DataFrame(rows); abl.to_csv(OUT / "program_vs_pca_ablation.csv", index=False)
    dl = pd.DataFrame(deltas); dl.to_csv(OUT / "program_vs_pca_paired_delta.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "program_vs_pca_calibration.csv", index=False)

    best_pca = abl[abl.feature_set == "clinical+cyto+RNA_PCA"].test_cindex.iloc[0]
    best_prog = abl[abl.feature_set == "clinical+cyto+miner3_programs"].test_cindex.iloc[0]
    verdict = ("RNA PCA >= miner3 programs" if best_pca >= best_prog else "miner3 programs > RNA PCA")
    card = [f"# Program-vs-PCA claim card (endpoint: open_gdc_os)", "",
            f"- Matched N={len(df)}, test {int(te.sum())} / {int(e.sum())} events. "
            f"Feature-count-matched: {NF} RNA PCs vs {NF} top-variance miner3 programs.",
            f"- clinical+cyto+RNA_PCA C={best_pca:.3f} vs clinical+cyto+miner3_programs C={best_prog:.3f} -> **{verdict}**.",
            "- All paired ΔC (programs vs PCA) and their CIs are in program_vs_pca_paired_delta.csv.",
            "- Endpoint = OS technical validation. NO relapse/PFS claim; NO clinical-use claim.",
            "- miner3 program activity is method-reproduced, not bit-validated (off-endpoint caveat applies)."]
    (OUT / "program_vs_pca_claim_card.md").write_text("\n".join(card) + "\n")

    print(abl.to_string(index=False))
    print("\nPaired ΔC (programs vs PCA):")
    print(dl[["comparison", "delta_cindex", "delta_ci_low", "delta_ci_high", "p_permutation", "claim"]].to_string(index=False))


if __name__ == "__main__":
    main()
