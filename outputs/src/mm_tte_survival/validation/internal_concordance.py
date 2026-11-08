"""Internal cross-modality concordance for the CoMMpass CNV subtype calls.

This is NOT FISH. It asks a weaker but fully-open question: do the copy-number-
derived subtype calls agree with the ORTHOGONAL RNA dosage signal they should
move with? A real del(17p) should depress TP53; a real amp(1q) should raise the
1q21 drivers (CKS1B, MCL1); del(13q) should depress RB1/DIS3; del(1p) should
depress the 1p32 tumour suppressors; hyperdiploidy should raise expression across
the trisomy chromosomes. Concordance here corroborates that the CNV caller tracks
biology; it cannot establish FISH-grade accuracy (for that see external_geo.py /
the literature table). Stamped NOT-FISH everywhere it is reported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .external_geo import binary_metrics, parse_annot
from .external_geo import HD_TRISOMY_CHROMS

# subtype -> (orthogonal marker genes, expected sign of expression vs the call).
# sign = -1 means a positive call should LOWER expression (deletion); +1 = raise.
MARKERS: Dict[str, Tuple[List[str], int]] = {
    "del17p": (["TP53"], -1),
    "amp1q": (["CKS1B", "MCL1", "ANP32E", "ILF2", "PDZK1IP1"], +1),
    "del13q": (["RB1", "DIS3"], -1),
    "del1p": (["CDKN2C", "FAF1"], -1),
    # hyperdiploid handled separately via the trisomy-chromosome signature.
}


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std().replace(0, np.nan) + 1e-9)


def gene_signature(log_expr: pd.DataFrame, genes: List[str], sign: int
                   ) -> Tuple[pd.Series, List[str]]:
    """sign * mean z-score across present marker genes (samples x genes -> series)."""
    present = [g for g in genes if g in log_expr.columns]
    if not present:
        return pd.Series(np.nan, index=log_expr.index), present
    return sign * _zscore(log_expr[present]).mean(axis=1), present


def trisomy_signature(log_expr: pd.DataFrame, gene_chrom: Dict[str, str],
                      chroms: List[str]) -> pd.Series:
    """Mean z across genes located on `chroms` (higher = more copies)."""
    genes = [g for g in log_expr.columns if gene_chrom.get(g) in chroms]
    if not genes:
        return pd.Series(np.nan, index=log_expr.index)
    return _zscore(log_expr[genes]).mean(axis=1)


def _pointbiserial(y: np.ndarray, s: np.ndarray) -> float:
    keep = ~(np.isnan(y) | np.isnan(s))
    if keep.sum() < 3 or len(set(y[keep])) < 2:
        return float("nan")
    return float(np.corrcoef(y[keep], s[keep])[0, 1])


def run_internal_concordance(gene_matrix_npz: str = "data/real/gene_matrix.npz",
                             cyto_csv: str = "data/real/cytogenetics.csv",
                             annot_gz: str = "data/external/GPL570.annot.gz",
                             outdir: str = "outputs/validation") -> pd.DataFrame:
    npz = np.load(gene_matrix_npz, allow_pickle=True)
    log_expr = pd.DataFrame(npz["matrix"], index=[str(p) for p in npz["patient_ids"]],
                            columns=[str(g) for g in npz["genes"]])
    cyto = pd.read_csv(cyto_csv).set_index("patient_id")
    cyto.index = cyto.index.astype(str)
    common = log_expr.index.intersection(cyto.index)
    log_expr = log_expr.loc[common]
    cyto = cyto.loc[common]

    rows = []
    for sub, (genes, sign) in MARKERS.items():
        if sub not in cyto.columns:
            continue
        sig, present = gene_signature(log_expr, genes, sign)
        y = cyto[sub]
        m = binary_metrics(y.values, sig.values)
        rows.append({"subtype": sub, "gold": "orthogonal_expression", "is_fish": False,
                     "markers": "+".join(present) or "ABSENT", "direction": "down" if sign < 0 else "up",
                     "pointbiserial_r": _pointbiserial(y.values.astype(float), sig.values), **m})

    # hyperdiploid via trisomy-chromosome dosage (gene->chrom from GPL570 annot).
    if "hyperdiploid" in cyto.columns and Path(annot_gz).exists():
        gene_chrom = (parse_annot(annot_gz).reset_index()
                      .drop_duplicates("symbol").set_index("symbol")["chrom"].to_dict())
        sig = trisomy_signature(log_expr, gene_chrom, HD_TRISOMY_CHROMS)
        y = cyto["hyperdiploid"]
        m = binary_metrics(y.values, sig.values)
        rows.append({"subtype": "hyperdiploid", "gold": "orthogonal_expression", "is_fish": False,
                     "markers": "tri(3,5,7,9,11,15,19,21)", "direction": "up",
                     "pointbiserial_r": _pointbiserial(y.values.astype(float), sig.values), **m})

    res = pd.DataFrame(rows)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "internal_concordance.csv", index=False)
    _write_card(res, out / "internal_concordance_card.md")
    return res


def _write_card(res: pd.DataFrame, path: Path) -> None:
    cols = ["subtype", "markers", "direction", "n", "n_pos", "auc", "pointbiserial_r", "kappa"]
    cols = [c for c in cols if c in res.columns]
    d = res[cols].copy()
    for c in ["auc", "pointbiserial_r", "kappa"]:
        if c in d:
            d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    lines = [
        "# Internal cross-modality concordance (CoMMpass) — NOT FISH",
        "",
        "Copy-number subtype calls vs the orthogonal RNA dosage signal they should",
        "track. Corroborates that the CNV caller follows biology; does NOT establish",
        "FISH-grade accuracy. AUC = how well the expression signature ranks the CNV",
        "call; pointbiserial_r = correlation of call with signature.",
        "",
        header, sep, *body,
    ]
    path.write_text("\n".join(lines) + "\n")
