"""Cytogenetic feature provenance helpers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_provenance(real_dir: str | Path) -> pd.DataFrame | None:
    p = Path(real_dir) / "cytogenetics_provenance.csv"
    return pd.read_csv(p) if p.exists() else None


def confirmatory_allowed(provenance: pd.DataFrame | None, feature: str) -> bool:
    """RNA-surrogate cytogenetics must never be treated as FISH/genomic calls."""
    if provenance is None:
        return False
    row = provenance[provenance["feature"] == feature]
    if row.empty:
        return False
    return bool(row["confirmatory_allowed"].iloc[0])
