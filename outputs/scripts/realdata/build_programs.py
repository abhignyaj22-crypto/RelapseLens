#!/usr/bin/env python
"""Step 3c (optional): build curated MM program-activity features from real RNA.

A biologically interpretable alternative/supplement to generic RNA PCA. Each
"program" activity = mean z-score (across the matched cohort) of a curated,
literature-based MM/oncology gene signature computed from log2(TPM+1). This is a
transparent stand-in for MINER/mmSYGNAL regulon activities (whose proprietary
regulon definitions are not available open-access); when those become available,
swap them in here.

Output: data/real/program_activity.csv  (patient_id + one column per program).
The matched-cohort ablation auto-detects this file and adds
clinical+programs / clinical+cytogenetics+programs arms.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_omics import parse_star  # reuse the STAR parser

ROOT = Path(__file__).resolve().parents[2]
RNADIR = ROOT / "data" / "real" / "rna_counts"
MAPCSV = ROOT / "data" / "real" / "rna_file_map.tsv"
OUT = ROOT / "data" / "real" / "program_activity.csv"

# Curated MM / oncology signatures (public gene sets; literature-based).
SIGNATURES = {
    "proliferation": ["MKI67", "TOP2A", "CCNB1", "CCNB2", "CDK1", "BUB1", "AURKA",
                      "MCM2", "MCM3", "MCM6", "PCNA", "TYMS", "TFDP1", "FOXM1", "RRM2"],
    "mmset_t4_14": ["NSD2", "FGFR3"],
    "ccnd1_t11_14": ["CCND1", "CCND2"],
    "maf_program": ["MAF", "MAFB", "ITGB7", "CCND2"],
    "myc_targets": ["MYC", "NPM1", "NCL", "PA2G4", "SRM", "ODC1"],
    "nfkb": ["NFKB1", "NFKB2", "RELB", "BIRC3", "TNFAIP3", "CD40"],
    "ifn_response": ["STAT1", "IRF1", "MX1", "OAS1", "ISG15", "IFIT1"],
    "plasma_cell_diff": ["PRDM1", "XBP1", "IRF4", "SDC1", "TNFRSF17"],
    "bone_marrow_stroma": ["DKK1", "FRZB", "WIF1"],
    "highrisk_gep_proxy": ["TBRG4", "FABP5", "PDHA1", "ENO1", "LDHA", "EXOSC4",
                           "KIF14", "CKS1B", "ASPM", "CTPS1"],
}


def main():
    fmap = pd.read_csv(MAPCSV, sep="\t")
    vectors, pids = {}, []
    for _, r in fmap.iterrows():
        path = RNADIR / f"{r['file_id']}.star.tsv"
        if not path.exists() or path.stat().st_size < 1000:
            continue
        vectors[r["patient_id"]] = parse_star(path)
        pids.append(r["patient_id"])
    if not pids:
        sys.exit("no RNA files parsed")
    mat = pd.DataFrame.from_dict(vectors, orient="index").sort_index()
    mat = mat.dropna(axis=1, how="any")
    log_tpm = np.log2(mat + 1.0)
    z = (log_tpm - log_tpm.mean()) / (log_tpm.std() + 1e-9)

    out = pd.DataFrame(index=log_tpm.index)
    for name, genes in SIGNATURES.items():
        present = [g for g in genes if g in z.columns]
        if not present:
            print(f"  WARN {name}: no marker genes present, skipping", file=sys.stderr)
            continue
        out[f"prog_{name}"] = z[present].mean(axis=1).values
        print(f"  prog_{name}: {len(present)}/{len(genes)} genes", file=sys.stderr)
    out.insert(0, "patient_id", out.index)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({out.shape[0]} patients x {out.shape[1]-1} programs)")


if __name__ == "__main__":
    main()
