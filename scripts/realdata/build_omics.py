#!/usr/bin/env python
"""Step 3b: build omics.csv (PC1..PC128) from real STAR gene-counts, and add
expression-surrogate IGH-translocation calls to cytogenetics.csv.

LEGACY / DIAGNOSTIC ONLY (leakage-risk): the PCA below is fit on the FULL cohort,
so omics.csv PCs leak test information. They are retained only as a labeled
diagnostic comparison. The leak-free path is data/gene_expression.build_gene_matrix
(scripts/realdata/build_gene_matrix.py) + data/omics_pca.OmicsInFoldPCA, which fits
gene-selection + scaling + PCA INSIDE each train fold. Likewise the t(*) surrogates
here are thresholded on full-cohort quantiles -> excluded from primary HSS membership
(see cohort.CNV_SUBTYPES).

Pipeline:
  1. Parse each STAR tsv -> protein-coding tpm_unstranded vector.
  2. Assemble patient x gene log2(TPM+1) matrix.
  3. Select top-2000 most-variable genes, z-score, PCA -> PC1..PC128.
  4. Translocation surrogates from canonical marker overexpression:
       t(11;14) -> CCND1, t(4;14) -> NSD2/WHSC1 (+FGFR3), t(14;16) -> MAF.
     Calls are expression SURROGATES, not FISH; thresholds set to literature
     prevalence and reported for transparency.

No synthetic data: every value derives from downloaded GDC RNA.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Single source of truth for the surrogate caller, shared with the external GEO
# validation so both label the same way (validation/surrogate_caller.py).
from mm_tte_survival.validation.surrogate_caller import TRANSLOC, call_translocation

ROOT = Path(__file__).resolve().parents[2]
RNADIR = ROOT / "data" / "real" / "rna_counts"
MAPCSV = ROOT / "data" / "real" / "rna_file_map.tsv"
OUT_OMICS = ROOT / "data" / "real" / "omics.csv"
CYTO = ROOT / "data" / "real" / "cytogenetics.csv"

N_PCS = 128
N_TOPVAR = 2000
SKIP_PREFIX = ("N_", "#")


def parse_star(path: Path):
    """Return dict gene_name -> tpm_unstranded for protein_coding genes."""
    out = {}
    with path.open() as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None and parts[0] == "gene_id":
                header = parts
                idx = {c: i for i, c in enumerate(header)}
                continue
            if header is None or parts[0].startswith(SKIP_PREFIX):
                continue
            try:
                gtype = parts[idx["gene_type"]]
                if gtype != "protein_coding":
                    continue
                name = parts[idx["gene_name"]]
                tpm = float(parts[idx["tpm_unstranded"]])
            except (KeyError, ValueError, IndexError):
                continue
            # collapse duplicate gene_names by max TPM
            if name not in out or tpm > out[name]:
                out[name] = tpm
    return out


def main():
    fmap = pd.read_csv(MAPCSV, sep="\t")
    vectors, pids = {}, []
    for _, r in fmap.iterrows():
        path = RNADIR / f"{r['file_id']}.star.tsv"
        if not path.exists() or path.stat().st_size < 1000:
            continue
        vectors[r["patient_id"]] = parse_star(path)
        pids.append(r["patient_id"])
    print(f"parsed {len(pids)} RNA samples", file=sys.stderr)
    if not pids:
        sys.exit("no RNA files parsed")

    mat = pd.DataFrame.from_dict(vectors, orient="index").sort_index()
    mat = mat.dropna(axis=1, how="any")  # genes present in all samples
    print(f"gene matrix: {mat.shape[0]} patients x {mat.shape[1]} protein-coding genes", file=sys.stderr)
    log_tpm = np.log2(mat + 1.0)

    # ---- expression-surrogate translocation calls (shared caller) ----
    surro = pd.DataFrame(index=log_tpm.index)
    for tcol, (genes, prev) in TRANSLOC.items():
        calls, _, present = call_translocation(log_tpm, genes, prev)
        surro[tcol] = calls
        if not present:
            print(f"  {tcol}: marker genes {genes} absent -> NA", file=sys.stderr)
        else:
            print(f"  {tcol}: markers={present} prevalence={np.nanmean(calls)*100:.1f}% "
                  f"(target {prev*100:.0f}%)", file=sys.stderr)

    # ---- PCA embedding ----
    variances = log_tpm.var(axis=0).sort_values(ascending=False)
    top_genes = variances.index[:N_TOPVAR]
    X = StandardScaler().fit_transform(log_tpm[top_genes].values)
    n_pcs = min(N_PCS, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_pcs, random_state=42).fit(X)
    Z = pca.transform(X)
    evr = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_pcs} PCs capture {evr*100:.1f}% variance of top-{N_TOPVAR} genes", file=sys.stderr)

    omics = pd.DataFrame(Z, index=log_tpm.index,
                         columns=[f"PC{i+1}" for i in range(n_pcs)])
    omics.insert(0, "patient_id", omics.index)
    omics.to_csv(OUT_OMICS, index=False)
    print(f"wrote {OUT_OMICS}  ({omics.shape[0]} x {n_pcs} PCs)")

    # gene x PC loadings (for PC->gene driver inspection / enrichment downstream)
    loadings = pd.DataFrame(pca.components_.T, index=top_genes,
                            columns=[f"PC{i+1}" for i in range(n_pcs)])
    loadings.to_csv(ROOT / "data" / "real" / "omics_pc_loadings.csv")

    # ---- cytogenetics provenance manifest (reviewers must not treat RNA
    #      surrogates as FISH/genomic calls) ----
    prov_rows = [
        ("amp1q", "WGS Copy Number Segment", "length-weighted 1q arm log2 (genome-median recentred)", "moderate", False),
        ("del1p", "WGS Copy Number Segment", "length-weighted 1p arm log2", "moderate", False),
        ("del13q", "WGS Copy Number Segment", "length-weighted chr13 log2", "moderate", False),
        ("del17p", "WGS Copy Number Segment", "length-weighted 17p arm log2 (subclonal-sensitive)", "moderate", False),
        ("hyperdiploid", "WGS Copy Number Segment", ">=2 trisomy of odd chromosomes", "moderate", False),
        ("t_4_14", "RNA expression", "NSD2/FGFR3 over-expression surrogate (z>cutoff @ prevalence)", "exploratory", False),
        ("t_11_14", "RNA expression", "CCND1 over-expression surrogate", "exploratory", False),
        ("t_14_16", "RNA expression", "MAF over-expression surrogate", "exploratory", False),
    ]
    prov = pd.DataFrame(prov_rows, columns=["feature", "source", "method", "confidence", "confirmatory_allowed"])
    prov.to_csv(ROOT / "data" / "real" / "cytogenetics_provenance.csv", index=False)
    print("wrote cytogenetics_provenance.csv (CNV calls + RNA-surrogate flags)")

    # ---- merge surrogate translocations into cytogenetics.csv ----
    if CYTO.exists():
        cyto = pd.read_csv(CYTO)
        s = surro.reset_index().rename(columns={"index": "patient_id"})
        for tcol in TRANSLOC:
            cyto = cyto.drop(columns=[tcol], errors="ignore")
        cyto = cyto.merge(s, on="patient_id", how="left")
        order = ["patient_id", "amp1q", "del1p", "del13q", "del17p",
                 "t_4_14", "t_11_14", "t_14_16", "hyperdiploid"]
        cyto = cyto[[c for c in order if c in cyto.columns]]
        cyto.to_csv(CYTO, index=False)
        print(f"updated {CYTO} with expression-surrogate translocations")


if __name__ == "__main__":
    main()
