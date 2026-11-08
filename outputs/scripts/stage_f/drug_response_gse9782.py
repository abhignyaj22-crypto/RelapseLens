#!/usr/bin/env python3
"""Stage F — bortezomib-response prediction on GSE9782 (external, drug-labeled).

A DIFFERENT endpoint from OS: does the cell-line drug foundation transfer to patient
drug response where it did not move survival? Bortezomib (PS-341) targets the 26S
proteasome, so we test whether a mechanism-informed feature set — proteasome-subunit
genes, and genes flagged by DepMap CRISPR as most essential (foundation-derived) —
predicts PGx_Responder (R/NR) better than a naive whole-transcriptome logistic model.

Public data only (GSE9782 Mulligan 2007; DepMap CRISPR). Honest 5-fold CV AUROC with a
permutation control; a null (mechanism prior doesn't help) is reportable.
"""
from __future__ import annotations

import gzip
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
GSE = ROOT / "data/external/drug_labeled/GSE9782/GSE9782-GPL96_series_matrix.txt.gz"
ANNOT = ROOT / "data/external/GPL96.annot.gz"
CRISPR = ROOT / "data/external/drug_response/CRISPRGeneEffect.csv"
OUT = ROOT / "outputs/stage_f"
SEED = 42

PROTEASOME = sorted(set(  # dedup: PSMB8/9/10 already covered by the PSMB1..10 range
    [f"PSMB{i}" for i in range(1, 11)] + [f"PSMA{i}" for i in range(1, 8)] +
    [f"PSMC{i}" for i in range(1, 7)] + [f"PSMD{i}" for i in range(1, 15)] +
    ["NFE2L1", "POMP", "ADRM1"]))


def load_gse9782():
    """Return (expr genes x samples, responder labels aligned)."""
    with gzip.open(GSE, "rt") as fh:
        text = fh.read()
    # labels
    resp = None
    for line in text.splitlines():
        if line.startswith("!Sample_characteristics") and "PGx_Responder" in line:
            vals = re.findall(r'"PGx_Responder = ([A-Za-z]+)"', line)
            resp = vals
            break
    # expression table
    start = text.index("!series_matrix_table_begin")
    end = text.index("!series_matrix_table_end")
    table = text[start:end].split("\n", 1)[1]
    df = pd.read_csv(io.StringIO(table), sep="\t", index_col=0)
    df = df[[c for c in df.columns if c.startswith("GSM")]]
    y = np.array([1 if r == "R" else 0 for r in resp])
    return df, y


def probe_to_gene() -> dict[str, str]:
    with gzip.open(ANNOT, "rt") as fh:
        lines = [l for l in fh if not l.startswith(("^", "!", "#"))]
    ann = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)
    idc = ann.columns[0]
    gc = next((c for c in ann.columns if "Gene symbol" in c or c == "Gene symbol"), None)
    if gc is None:
        return {}
    return {str(p): str(g).split("///")[0].strip() for p, g in zip(ann[idc], ann[gc]) if pd.notna(g)}


def collapse_genes(expr: pd.DataFrame, p2g: dict) -> pd.DataFrame:
    genes = [p2g.get(str(p)) for p in expr.index]
    expr = expr.copy()
    expr.index = genes
    valid = [isinstance(g, str) and g != "" and g != "nan" for g in genes]
    expr = expr.loc[valid]                       # boolean ROW mask
    return expr.apply(pd.to_numeric, errors="coerce").groupby(level=0).mean()  # gene x sample


def cv_auroc(X, y, seed=SEED):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, proba)


def main() -> int:
    expr, y = load_gse9782()
    genes = collapse_genes(expr, probe_to_gene())          # gene x sample
    X_all = genes.T.values                                  # samples x genes
    samples_genes = list(genes.index)
    print(f"[gse9782] samples={X_all.shape[0]} responders={int(y.sum())}/{len(y)} genes={X_all.shape[1]}", flush=True)

    # feature sets
    prot_idx = [i for i, g in enumerate(samples_genes) if g in set(PROTEASOME)]
    # foundation prior: top-500 most-essential genes by DepMap CRISPR (most negative mean effect)
    cr = pd.read_csv(CRISPR, index_col=0)
    ess = cr.mean(axis=0)
    ess.index = [re.sub(r"\s*\(\d+\)$", "", c) for c in ess.index]
    top_ess = set(ess.sort_values().index[:500])
    ess_idx = [i for i, g in enumerate(samples_genes) if g in top_ess]

    rng = np.random.default_rng(SEED)
    res = {
        "n_samples": int(X_all.shape[0]), "n_responders": int(y.sum()), "n_genes": X_all.shape[1],
        "auroc_full_transcriptome": float(cv_auroc(X_all, y)),
        "auroc_proteasome_signature": (float(cv_auroc(X_all[:, prot_idx], y)) if len(prot_idx) >= 3 else None),
        "n_proteasome_genes": len(prot_idx),
        "auroc_crispr_essential_top500": (float(cv_auroc(X_all[:, ess_idx], y)) if len(ess_idx) >= 10 else None),
        "n_crispr_essential_genes": len(ess_idx),
        "auroc_permuted_labels_control": float(cv_auroc(X_all, rng.permutation(y))),
        "baseline_note": "AUROC 0.5 = chance; published GSE9782 bortezomib signatures ~0.6-0.65",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drug_response_gse9782.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
