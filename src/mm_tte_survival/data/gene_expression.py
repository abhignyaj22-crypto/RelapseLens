"""Raw gene-expression substrate for leak-free, in-fold PCA.

`build_omics.py` fits gene-selection + scaling + PCA on the FULL cohort (a leak).
This module caches a FOLD-AGNOSTIC patient x gene log2(TPM+1) matrix (no selection,
no scaling, no PCA) so that `OmicsInFoldPCA` can fit everything inside the train
fold. The cache is large and rebuildable -> gitignored.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

_SKIP_PREFIX = ("N_", "#")


def parse_star(path: Path) -> dict:
    """gene_name -> tpm_unstranded for protein_coding genes (max-collapse dups).
    Identical parsing to scripts/realdata/build_omics.py (single source of truth)."""
    out: dict = {}
    with path.open() as f:
        header = None
        idx = {}
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None and parts[0] == "gene_id":
                header = parts
                idx = {c: i for i, c in enumerate(header)}
                continue
            if header is None or parts[0].startswith(_SKIP_PREFIX):
                continue
            try:
                if parts[idx["gene_type"]] != "protein_coding":
                    continue
                name = parts[idx["gene_name"]]
                tpm = float(parts[idx["tpm_unstranded"]])
            except (KeyError, ValueError, IndexError):
                continue
            if name not in out or tpm > out[name]:
                out[name] = tpm
    return out


def build_gene_matrix(rna_dir: Path, file_map: Path, out_path: Path) -> Path:
    """Parse every STAR file -> patient x gene log2(TPM+1) matrix -> .npz cache.
    Genes are kept only if present in ALL samples (fold-agnostic filter)."""
    fmap = pd.read_csv(file_map, sep="\t")
    vectors, pids = {}, []
    for _, r in fmap.iterrows():
        path = rna_dir / f"{r['file_id']}.star.tsv"
        if not path.exists() or path.stat().st_size < 1000:
            continue
        vectors[str(r["patient_id"])] = parse_star(path)
        pids.append(str(r["patient_id"]))
    if not pids:
        raise SystemExit("no RNA files parsed")
    mat = pd.DataFrame.from_dict(vectors, orient="index").sort_index()
    mat = mat.dropna(axis=1, how="any")
    log_tpm = np.log2(mat.values.astype("float32") + 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, matrix=log_tpm,
                        patient_ids=np.array(mat.index, dtype=object),
                        genes=np.array(mat.columns, dtype=object))
    print(f"wrote {out_path}  ({log_tpm.shape[0]} patients x {log_tpm.shape[1]} genes)", file=sys.stderr)
    return out_path


def load_gene_matrix(path: Path) -> pd.DataFrame:
    """Load the cache as a patient_id-indexed DataFrame of log2(TPM+1)."""
    z = np.load(path, allow_pickle=True)
    return pd.DataFrame(z["matrix"], index=[str(p) for p in z["patient_ids"]],
                        columns=[str(g) for g in z["genes"]])
