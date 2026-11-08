"""External validation of the sequencing-inferred subtype labels on open GEO
cohorts that carry REAL FISH (or a published expression-cluster gold standard).

Two open datasets, two honest roles (see docs/subtype_label_validation.md):

  * GSE6477  (n~162, Affymetrix U133A, Chng et al.)  -> genuine interphase FISH
    for del(13) and hyperdiploidy. Expression-only platform, so this validates an
    *expression* detector for those two CNV subtypes against real FISH, on a
    different array than CoMMpass (cross-platform caveat stated).

  * GSE19784 (n~328, U133 Plus 2.0, HOVON-65/GMMG-HD4, Broyl et al.) -> the
    published molecular CLUSTER (MS=t(4;14), MF=t(14;16), CD-1/CD-2=t(11;14)).
    The cluster is transcriptome-derived, so this is cluster-CONCORDANCE for the
    translocation surrogates, NOT raw FISH. It tests whether our minimal
    1-3-gene caller (validation/surrogate_caller.py) recovers the established
    full-transcriptome classification.

No FISH for del(17p)/amp(1q)/del(1p) exists in either set; those rely on the
literature concordance table (docs/literature_cnv_fish_concordance.md). Nothing
here is synthetic: every value derives from the downloaded public GEO records.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .surrogate_caller import TRANSLOC, call_translocation

# Canonical hyperdiploid trisomies in MM (odd chromosomes).
HD_TRISOMY_CHROMS = ["3", "5", "7", "9", "11", "15", "19", "21"]

# Series-matrix + platform-annotation FTP sources (real public data).
GEO = {
    "GSE6477": {
        "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE6nnn/GSE6477/matrix/GSE6477_series_matrix.txt.gz",
        "platform": "GPL96",
    },
    "GSE19784": {
        "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19784/matrix/GSE19784_series_matrix.txt.gz",
        "platform": "GPL570",
    },
}
ANNOT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/{gpl}/annot/{gpl}.annot.gz"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def parse_series_matrix(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (expr [probe x sample], characteristics [sample x char_i]).

    characteristics columns are the successive !Sample_characteristics_ch1 rows,
    values are the raw 'key: value' strings (one per sample).
    """
    path = Path(path)
    char_rows: List[List[str]] = []
    sample_ids: Optional[List[str]] = None
    rows, idx = [], []
    in_table = False
    with _open(path) as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                parts = line.rstrip("\n").split("\t")
                vals = [p.strip().strip('"') for p in parts]
                if vals[0].upper() == "ID_REF":
                    sample_ids = vals[1:]
                    continue
                idx.append(vals[0])
                rows.append([_num(v) for v in vals[1:]])
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = line.rstrip("\n").split("\t")
                char_rows.append([p.strip().strip('"') for p in parts[1:]])
    if sample_ids is None:
        raise ValueError(f"No expression table found in {path}")
    expr = pd.DataFrame(rows, index=idx, columns=sample_ids).apply(pd.to_numeric, errors="coerce")
    chars = pd.DataFrame({f"char{i}": r for i, r in enumerate(char_rows)}, index=sample_ids)
    return expr, chars


def _num(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def parse_annot(path: Path) -> pd.DataFrame:
    """Return probe -> (symbol, chrom) from a GEO GPL .annot.gz file.
    Columns of interest: 'ID', 'Gene symbol', 'Chromosome location'."""
    path = Path(path)
    header, rows = None, []
    with _open(path) as f:
        for line in f:
            if line.startswith(("^", "!", "#")):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                if parts[0] == "ID":
                    header = parts
                    ci = {c: i for i, c in enumerate(header)}
                continue
            try:
                probe = parts[ci["ID"]]
                sym = parts[ci["Gene symbol"]]
                loc = parts[ci["Chromosome location"]] if "Chromosome location" in ci else ""
            except (KeyError, IndexError):
                continue
            rows.append((probe, sym, _chrom_of(loc)))
    return pd.DataFrame(rows, columns=["probe", "symbol", "chrom"]).set_index("probe")


def _chrom_of(loc: str) -> str:
    """'13q14.2' -> '13'; '' -> ''. Takes the first arm token only."""
    if not loc:
        return ""
    m = re.match(r"\s*([0-9]{1,2}|X|Y)", loc.split("|")[0].split("//")[0])
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- #
# Expression assembly
# --------------------------------------------------------------------------- #
def to_log(expr: pd.DataFrame) -> pd.DataFrame:
    """log2(x+1) if the array is on a linear scale (MAS5); pass through if already
    log (RMA). Heuristic: 99th percentile > 50 => linear."""
    p99 = np.nanpercentile(expr.values, 99)
    return np.log2(expr.clip(lower=0) + 1.0) if p99 > 50 else expr


def gene_matrix(expr_log: pd.DataFrame, annot: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    """sample x gene log-expression, collapsing multiple probes per gene by MAX
    (matches the max-collapse used when the CoMMpass cohort was built)."""
    out = {}
    for g in genes:
        probes = annot.index[annot["symbol"] == g]
        probes = [p for p in probes if p in expr_log.index]
        if probes:
            out[g] = expr_log.loc[probes].max(axis=0)
    return pd.DataFrame(out)  # index = samples


def chrom_signature(expr_log: pd.DataFrame, annot: pd.DataFrame, chroms: List[str]) -> pd.Series:
    """Mean per-gene z-score across all probes on `chroms` (higher = more
    expression / more copies). Used as a copy-number dosage proxy."""
    probes = [p for p in annot.index[annot["chrom"].isin(chroms)] if p in expr_log.index]
    if not probes:
        return pd.Series(np.nan, index=expr_log.columns)
    sub = expr_log.loc[probes]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan) + 1e-9, axis=0)
    return z.mean(axis=0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y_true: np.ndarray, score: np.ndarray, prevalence: Optional[float] = None,
                   threshold: Optional[float] = None) -> Dict[str, float]:
    """AUC + confusion-derived sens/spec/PPV/NPV/kappa at a threshold. If
    `threshold` is None, threshold the score at its (1 - prevalence) quantile,
    where prevalence defaults to the observed positive rate of y_true."""
    from sklearn.metrics import roc_auc_score, cohen_kappa_score

    y = np.asarray(y_true, float)
    s = np.asarray(score, float)
    keep = ~(np.isnan(y) | np.isnan(s))
    y, s = y[keep], s[keep]
    n = int(keep.sum())
    out = {"n": n, "n_pos": int(y.sum()), "auc": np.nan, "sens": np.nan, "spec": np.nan,
           "ppv": np.nan, "npv": np.nan, "kappa": np.nan, "threshold": np.nan}
    if n == 0 or y.sum() == 0 or y.sum() == n:
        return out
    out["auc"] = float(roc_auc_score(y, s))
    if threshold is None:
        prev = float(y.mean()) if prevalence is None else float(prevalence)
        threshold = float(np.quantile(s, 1.0 - prev))
    out["threshold"] = float(threshold)
    pred = (s > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    out["sens"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["spec"] = tn / (tn + fp) if (tn + fp) else np.nan
    out["ppv"] = tp / (tp + fp) if (tp + fp) else np.nan
    out["npv"] = tn / (tn + fn) if (tn + fn) else np.nan
    out["kappa"] = float(cohen_kappa_score(y, pred)) if len(set(pred)) > 1 else 0.0
    return out


# --------------------------------------------------------------------------- #
# Dataset-specific validations
# --------------------------------------------------------------------------- #
def validate_gse19784(expr: pd.DataFrame, chars: pd.DataFrame, annot: pd.DataFrame) -> pd.DataFrame:
    """Translocation surrogate caller vs published molecular cluster (NOT FISH)."""
    cluster_pos = {"t_4_14": {"MS"}, "t_14_16": {"MF"}, "t_11_14": {"CD-1", "CD-2"}}
    # locate the cluster characteristics column
    cl_col = _find_char(chars, "cluster:")
    cluster = chars[cl_col].str.replace("cluster:", "", regex=False).str.strip() if cl_col else None
    expr_log = to_log(expr)
    rows = []
    for tcol, (genes, prev) in TRANSLOC.items():
        gm = gene_matrix(expr_log, annot, genes)
        if gm.empty or cluster is None:
            rows.append({"dataset": "GSE19784", "subtype": tcol, "gold": "molecular_cluster",
                         "is_fish": False, "n": 0, "note": "markers or cluster absent"})
            continue
        calls, score, present = call_translocation(gm, list(gm.columns), prev)
        y = cluster.isin(cluster_pos[tcol]).astype(int).reindex(score.index)
        m = binary_metrics(y.values, score.values)
        rows.append({"dataset": "GSE19784", "subtype": tcol, "gold": "molecular_cluster",
                     "is_fish": False, "markers": "+".join(present),
                     "cluster_prev": float(y.mean()), **m})
    return pd.DataFrame(rows)


def validate_gse6477(expr: pd.DataFrame, chars: pd.DataFrame, annot: pd.DataFrame) -> pd.DataFrame:
    """Real FISH del(13) + hyperdiploidy vs expression dosage signatures."""
    expr_log = to_log(expr)
    rows = []

    # del(13): FISH column contains 'Chromosome 13'; signature = -mean expr on chr13.
    c13 = _find_char(chars, "Chromosome 13")
    if c13:
        y = chars[c13].str.contains("deletion", case=False, na=False).astype(int)
        sig = -chrom_signature(expr_log, annot, ["13"]).reindex(y.index)
        m = binary_metrics(y.values, sig.values)
        rows.append({"dataset": "GSE6477", "subtype": "del13q", "gold": "FISH", "is_fish": True,
                     "signature": "-mean_expr(chr13)", "fish_prev": float(y.mean()), **m})

    # hyperdiploidy: FISH 'Hyperdiploid'; signature = mean expr on trisomy chroms.
    chd = _find_char(chars, "Hyperdiploid")
    if chd:
        y = (chars[chd].str.strip().str.lower() == "hyperdiploid").astype(int)
        sig = chrom_signature(expr_log, annot, HD_TRISOMY_CHROMS).reindex(y.index)
        m = binary_metrics(y.values, sig.values)
        rows.append({"dataset": "GSE6477", "subtype": "hyperdiploid", "gold": "FISH", "is_fish": True,
                     "signature": "mean_expr(tri 3,5,7,9,11,15,19,21)", "fish_prev": float(y.mean()), **m})
    return pd.DataFrame(rows)


def _find_char(chars: pd.DataFrame, needle: str) -> Optional[str]:
    """Return the first characteristics column whose values mention `needle`."""
    for c in chars.columns:
        if chars[c].astype(str).str.contains(needle, case=False, na=False).any():
            return c
    return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def ensure_cached(gse: str, cache_dir: Path) -> Tuple[Path, Path]:
    """Return (matrix_path, annot_path), downloading from GEO FTP if absent."""
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpath = cache_dir / f"{gse}_series_matrix.txt.gz"
    gpl = GEO[gse]["platform"]
    apath = cache_dir / f"{gpl}.annot.gz"
    if not mpath.exists():
        urllib.request.urlretrieve(GEO[gse]["matrix"], mpath)
    if not apath.exists():
        urllib.request.urlretrieve(ANNOT_URL.format(gpl=gpl), apath)
    return mpath, apath


def run_external_validation(cache_dir: str = "data/external",
                            outdir: str = "outputs/validation") -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for gse, fn in [("GSE6477", validate_gse6477), ("GSE19784", validate_gse19784)]:
        mpath, apath = ensure_cached(gse, cache_dir)
        expr, chars = parse_series_matrix(mpath)
        annot = parse_annot(apath)
        frames.append(fn(expr, chars, annot))
    res = pd.concat(frames, ignore_index=True)
    csv = out / "external_geo.csv"
    res.to_csv(csv, index=False)
    _write_card(res, out / "external_geo_card.md")
    return res


def _write_card(res: pd.DataFrame, path: Path) -> None:
    fish = res[res["is_fish"] == True]  # noqa: E712
    clust = res[res["is_fish"] == False]  # noqa: E712
    lines = [
        "# External subtype-label validation (open GEO)",
        "",
        "Real FISH ground truth exists on open data only for del(13) and hyperdiploidy",
        "(GSE6477); translocations are validated against the published expression",
        "*cluster* (GSE19784), which is concordance, NOT raw FISH. del(17p)/amp(1q)/",
        "del(1p) have no open FISH and rely on docs/literature_cnv_fish_concordance.md.",
        "",
        "## Real-FISH validation (GSE6477)",
        _fmt(fish),
        "",
        "## Expression-cluster concordance (GSE19784, NOT FISH)",
        _fmt(clust),
    ]
    path.write_text("\n".join(lines) + "\n")


def _fmt(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no rows_"
    cols = [c for c in ["dataset", "subtype", "gold", "n", "n_pos", "auc", "sens", "spec",
                        "ppv", "kappa"] if c in df.columns]
    d = df[cols].copy()
    for c in ["auc", "sens", "spec", "ppv", "kappa"]:
        if c in d:
            d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([header, sep, *body])
