"""Single source of truth for the expression-surrogate translocation caller.

The production build (scripts/realdata/build_omics.py) and the external GEO
validation (validation/external_geo.py) MUST call the same function, otherwise the
external sens/spec numbers would validate a *different* caller than the one that
labelled the cohort. The logic is exactly the original build_omics.py lines:
z-score each marker over the cohort, take the max across markers as the
translocation score, threshold at the (1 - prevalence) quantile.

These are EXPRESSION SURROGATES, not FISH.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Translocation marker genes and approximate MM prevalence (percentile cutoff).
# NSD2 and WHSC1 are the same gene (HGNC: NSD2, alias WHSC1); both aliases kept so
# the caller works whether a platform/annotation uses the old or new symbol.
TRANSLOC: Dict[str, Tuple[List[str], float]] = {
    "t_11_14": (["CCND1"], 0.16),
    "t_4_14": (["NSD2", "WHSC1", "FGFR3"], 0.13),
    "t_14_16": (["MAF"], 0.05),
}


def translocation_score(log_expr: pd.DataFrame, genes: List[str]) -> Tuple[pd.Series, List[str]]:
    """Cohort-standardized over-expression score = max z across present markers.

    log_expr: samples x genes, already on a log scale (log2(TPM+1) for RNA-seq,
    or log2 microarray intensity). Returns (score per sample, markers used).
    """
    present = [g for g in genes if g in log_expr.columns]
    if not present:
        return pd.Series(np.nan, index=log_expr.index), present
    sub = log_expr[present]
    z = (sub - sub.mean()) / (sub.std() + 1e-9)
    return z.max(axis=1), present


def call_translocation(log_expr: pd.DataFrame, genes: List[str], prevalence: float
                       ) -> Tuple[pd.Series, pd.Series, List[str]]:
    """Binary surrogate call at the (1 - prevalence) quantile of the score.

    Returns (calls 0/1, continuous score, markers used). Identical math to the
    original build_omics.py so external validation tests the deployed caller.
    """
    score, present = translocation_score(log_expr, genes)
    if not present:
        return pd.Series(np.nan, index=log_expr.index), score, present
    cutoff = score.quantile(1.0 - prevalence)
    calls = (score > cutoff).astype(int)
    return calls, score, present


def call_all_translocations(log_expr: pd.DataFrame,
                            transloc: Dict[str, Tuple[List[str], float]] = None
                            ) -> pd.DataFrame:
    """Call every translocation in `transloc` (default TRANSLOC). Columns are the
    translocation names; index matches log_expr."""
    transloc = transloc or TRANSLOC
    out = pd.DataFrame(index=log_expr.index)
    for tcol, (genes, prev) in transloc.items():
        calls, _, _ = call_translocation(log_expr, genes, prev)
        out[tcol] = calls
    return out
