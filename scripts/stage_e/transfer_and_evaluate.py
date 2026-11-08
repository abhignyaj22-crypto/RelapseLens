#!/usr/bin/env python3
"""Stage E — cell-line→patient transfer + honest survival value test.

Transcriptome-mediated transfer (Bareinboim–Pearl transportable): learn RNA→proteome-
latent on CCLE cell lines (which have BOTH), apply to patient CoMMpass RNA to get a
transferred multi-omic latent (never a patient measurement). Then test whether that
transferred latent adds discrimination over a clinical-only Cox for OS — the needle-
moving criterion (paired ΔC, CI>0). A null (transfer doesn't beat clinical/RNA) is a
legitimate, pre-registered outcome → demote transferred omics to explanatory.

Public data only. Reports: bridge quality (RNA→latent val R²), Uno-IPCW C-index for
{clinical, clinical+transferred, clinical+RNA-PCA}, paired ΔC + 95% CI over repeated
patient-disjoint splits.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DERIVED = ROOT / "data/external/_derived"
REAL = ROOT / "data/real"
OUT = ROOT / "outputs/stage_e"
SEED = 42
CLIN_COLS = ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy", "albumin", "b2m"]


def load_ccle_rna_named() -> pd.DataFrame:
    """CCLE RNA -> index CCLEName, columns gene symbol (TPM logp1)."""
    rna = pd.read_csv(DERIVED / "CCLE_OmicsExpressionTPM.csv", index_col=0)
    rna.columns = [re.sub(r"\s*\(\d+\)$", "", c) for c in rna.columns]
    model = pd.read_csv(ROOT / "data/external/drug_response/Model.csv")
    id2name = dict(zip(model["ModelID"], model["CCLEName"]))
    rna.index = [id2name.get(i, None) for i in rna.index]
    return rna[~rna.index.isna() & ~rna.index.duplicated()]


def build_transfer(shared_genes, dev):
    """RNA->proteome-latent bridge on cell lines, PROPERLY regularized to avoid the
    n<<p overfit trap (364 lines, ~19k genes): standardize -> PCA(50) -> ridge-CV to the
    32-dim latent. Returns the fitted (scaler, pca, ridge) + honest LOO/CV R²."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score

    rna = load_ccle_rna_named()
    lat = np.load(ROOT / "outputs/stage_d/ccle_latent.npz", allow_pickle=True)
    lat_df = pd.DataFrame(lat["latent"], index=lat["cell_lines"])
    lines = sorted(set(rna.index) & set(lat_df.index))
    genes = [g for g in shared_genes if g in rna.columns]
    Xc = rna.loc[lines, genes].fillna(0.0).values.astype("float32")
    Yc = lat_df.loc[lines].values.astype("float32")
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
    Xz = (Xc - mu) / sd

    pca = PCA(n_components=50, random_state=SEED).fit(Xz)
    Xp = pca.transform(Xz)
    ridge = RidgeCV(alphas=np.logspace(-1, 4, 12))
    # honest 5-fold CV R² (multioutput averaged) — the real transfer feasibility number
    cvpred = cross_val_predict(ridge, Xp, Yc, cv=5)
    cv_r2 = float(r2_score(Yc, cvpred))
    ridge.fit(Xp, Yc)
    return (mu, sd, pca, ridge), genes, cv_r2, len(lines)


def transfer_patient(bridge, genes, dev):
    mu, sd, pca, ridge = bridge
    d = np.load(REAL / "gene_matrix.npz", allow_pickle=True)
    gmat = pd.DataFrame(d["matrix"], index=[str(p) for p in d["patient_ids"]], columns=d["genes"])
    X = gmat.reindex(columns=genes).fillna(0.0).values.astype("float32")
    Xz = (X - mu) / sd                      # standardize with CCLE stats (domain shift is the test)
    z = ridge.predict(pca.transform(Xz))
    return pd.DataFrame(z, index=gmat.index).add_prefix("tz_")


def cox_cindex(train, test, cols, tcol, ecol):
    """Penalised-Cox Uno IPCW C-index (shared metric, no hand-rolled copy)."""
    from lifelines import CoxPHFitter
    from mm_tte_survival.metrics_ipcw import uno_c_index
    df = train[cols + [tcol, ecol]].apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median()).fillna(0.0)
    df[tcol] = train[tcol].clip(lower=0.1).values; df[ecol] = train[ecol].astype(int).values
    cph = CoxPHFitter(penalizer=0.1).fit(df, duration_col=tcol, event_col=ecol)
    te = test[cols].apply(pd.to_numeric, errors="coerce").fillna(df[cols].median()).fillna(0.0)
    risk = cph.predict_log_partial_hazard(te).values
    return uno_c_index(train[tcol].clip(lower=0.1).values, train[ecol].astype(int).values,
                       test[tcol].clip(lower=0.1).values, test[ecol].astype(int).values, risk)


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    clin = pd.read_csv(REAL / "clinical_survival.csv")
    clin["patient_id"] = clin["patient_id"].astype(str)

    patient_genes = list(np.load(REAL / "gene_matrix.npz", allow_pickle=True)["genes"])
    bridge, genes, bridge_r2, n_lines = build_transfer(patient_genes, dev)
    tz = transfer_patient(bridge, genes, dev)
    OUT.mkdir(parents=True, exist_ok=True)
    tz.to_csv(OUT / "patient_transferred_latent.csv")  # consumed by the fusion model

    # patient RNA — raw substrate for the "molecular alone" comparator. PCA is fit
    # IN-FOLD (train-only) per split via OmicsInFoldPCA; NOT a full-cohort fit (which
    # would leak test covariance into the comparator and inflate the transfer-vs-RNA gate).
    from mm_tte_survival.data.omics_pca import OmicsInFoldPCA
    d = np.load(REAL / "gene_matrix.npz", allow_pickle=True)
    gene_df = pd.DataFrame(d["matrix"], index=[str(p) for p in d["patient_ids"]], columns=list(d["genes"]))
    gene_df = gene_df.loc[:, gene_df.var(axis=0) > 0]            # drop constant genes

    df = clin.merge(tz, left_on="patient_id", right_index=True)
    df = df[df["patient_id"].isin(gene_df.index)]
    df = df[df["time_months"].notna() & df["event"].notna()].reset_index(drop=True)
    tzc = list(tz.columns)

    R = 25
    dC_transfer, dC_pca, C_clin_all = [], [], []
    for r in range(R):
        rs = np.random.default_rng(SEED + r)
        ix = rs.permutation(len(df)); cut = int(0.7 * len(df))
        train_ids = df.iloc[ix[:cut]]["patient_id"].tolist()
        pca = OmicsInFoldPCA(k=16).fit(gene_df, train_ids)      # train-only fit
        pc = pca.transform(gene_df.loc[df["patient_id"].values]).add_prefix("pc_")
        pc.index = df.index
        fdf = pd.concat([df, pc], axis=1)
        pcc = list(pc.columns)
        tr, te = fdf.iloc[ix[:cut]], fdf.iloc[ix[cut:]]
        if te["event"].sum() < 10:
            continue
        c_clin = cox_cindex(tr, te, CLIN_COLS, "time_months", "event")
        c_tz = cox_cindex(tr, te, CLIN_COLS + tzc, "time_months", "event")
        c_pca = cox_cindex(tr, te, CLIN_COLS + pcc, "time_months", "event")
        C_clin_all.append(c_clin); dC_transfer.append(c_tz - c_clin); dC_pca.append(c_pca - c_clin)

    def ci(a):
        a = np.array(a); return [float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    res = {
        "n_bridge_cell_lines": n_lines, "n_transfer_genes": len(genes),
        "bridge_rna_to_latent_cv_r2": bridge_r2,
        "n_patients_eval": int(len(df)), "n_splits": len(C_clin_all),
        "C_clinical_mean": float(np.mean(C_clin_all)),
        "dC_transfer_vs_clinical_mean_ci": ci(dC_transfer),
        "dC_rnaPCA_vs_clinical_mean_ci": ci(dC_pca),
        "gate_transfer_beats_clinical": bool(ci(dC_transfer)[1] > 0),
        "gate_transfer_beats_rnaPCA": bool(np.mean(dC_transfer) > np.mean(dC_pca)),
        "note": "ΔC CI excludes 0 => transfer adds discrimination; else pre-registered NULL (demote to explanatory).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "transfer_value.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
