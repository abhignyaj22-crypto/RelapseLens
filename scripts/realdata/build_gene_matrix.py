#!/usr/bin/env python
"""Cache the fold-agnostic patient x gene log2(TPM+1) matrix for in-fold PCA.

One-time substrate for the leak-free Stage-A re-baseline: NO gene selection,
NO scaling, NO PCA here (all of that happens inside the train fold via
OmicsInFoldPCA). Output is large and rebuildable -> gitignored.

  python scripts/realdata/build_gene_matrix.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mm_tte_survival.data.gene_expression import build_gene_matrix

if __name__ == "__main__":
    build_gene_matrix(
        rna_dir=ROOT / "data" / "real" / "rna_counts",
        file_map=ROOT / "data" / "real" / "rna_file_map.tsv",
        out_path=ROOT / "data" / "real" / "gene_matrix.npz",
    )
