#!/usr/bin/env python3
"""Stage D prep — build the paired CCLE proteome→epigenome training matrices.

Real public cell-line data (no synthetic). Aligns CCLE proteomics (Nusinow 2020,
proteins × per-cell-line TMT) with CCLE histone-PTM global chromatin profiling
(cell_line × 42 marks) on the shared cell-line axis. Output feeds CellLineVAE (L1).

Outputs data/external/_derived/ccle_paired.npz:
    proteome   (n_lines, n_proteins) float32   — VAE input
    epigenome  (n_lines, n_marks)    float32   — VAE reconstruction target
    cell_lines (n_lines,)   object              — CCLE names
    proteins   (n_proteins,) object             — Gene_Symbol
    marks      (n_marks,)    object             — histone marks
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROT = ROOT / "data/external/proteomics_cellline/protein_quant_current_normalized.csv.gz"
HIST = ROOT / "data/external/epigenomics_cellline/CCLE_GlobalChromatinProfiling_20181130.csv"
OUT = ROOT / "data/external/_derived/ccle_paired.npz"

MIN_PROTEIN_COVERAGE = 0.5   # keep proteins measured in >=50% of paired cell lines


def load_proteome() -> pd.DataFrame:
    """cell_line × protein (Gene_Symbol) abundance, averaging TMT columns per line."""
    df = pd.read_csv(PROT)
    quant_cols = [c for c in df.columns if "_TenPx" in c and not c.endswith("_Peptides")]
    line_of = {c: re.sub(r"_TenPx\d+.*$", "", c) for c in quant_cols}
    # proteins × quant-cols -> group quant cols by cell line (mean over TMT batches)
    mat = df[quant_cols].copy()
    mat.columns = [line_of[c] for c in quant_cols]
    mat = mat.T.groupby(level=0).mean().T           # proteins × cell_lines (deduped)
    mat.index = df["Gene_Symbol"].values
    mat = mat.groupby(level=0).mean()               # dedupe duplicate gene symbols
    return mat.T                                     # cell_line × protein


def load_epigenome() -> pd.DataFrame:
    df = pd.read_csv(HIST)
    marks = [c for c in df.columns if c not in ("CellLineName", "BroadID")]
    out = df.set_index("CellLineName")[marks]
    return out[~out.index.duplicated(keep="first")]


def main() -> int:
    prot = load_proteome()
    epi = load_epigenome()
    shared = sorted(set(prot.index) & set(epi.index))
    if len(shared) < 50:
        raise RuntimeError(f"too few paired cell lines: {len(shared)}")
    prot = prot.loc[shared]
    epi = epi.loc[shared]

    # keep well-covered proteins; impute remaining NaN by column mean (train-time stats)
    cov = prot.notna().mean(axis=0)
    prot = prot.loc[:, cov >= MIN_PROTEIN_COVERAGE]
    prot = prot.fillna(prot.mean(axis=0)).fillna(0.0)
    epi = epi.fillna(epi.mean(axis=0)).fillna(0.0)

    # Emit RAW (mean-imputed) matrices — standardization is deferred to train_ccle_vae,
    # which fits mean/std on the TRAIN split only (leak-free). Standardizing here on the
    # full cell-line set would leak val-cell-line scale into the held-out VAE metric.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        proteome=prot.values.astype("float32"),
        epigenome=epi.values.astype("float32"),
        cell_lines=np.array(shared, dtype=object),
        proteins=np.array(prot.columns, dtype=object),
        marks=np.array(epi.columns, dtype=object),
    )
    print(f"[prep_ccle] paired lines={len(shared)} proteins={prot.shape[1]} marks={epi.shape[1]}")
    print(f"[prep_ccle] wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
