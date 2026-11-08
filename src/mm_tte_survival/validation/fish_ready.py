"""FISH-ready harness — the future upgrade slot.

If a real CoMMpass FISH file is ever supplied (MMRF Researcher Gateway / dbGaP,
controlled-access), drop its path into the config and this computes genuine
per-subtype sensitivity / specificity / PPV / NPV / Cohen's kappa of the
sequencing-inferred calls against FISH. Until then it is inert (returns an empty
frame and a note), so the validation stack never fabricates a FISH comparison.

Expected FISH file: a CSV with a patient_id column and one 0/1 column per subtype
whose name matches the CoMMpass call columns (amp1q, del1p, del13q, del17p,
hyperdiploid, ...). Only columns present in BOTH files are scored.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def binary_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion-derived metrics for binary call vs binary FISH ground truth."""
    from sklearn.metrics import cohen_kappa_score

    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    keep = ~(np.isnan(y) | np.isnan(p))
    y, p = y[keep], p[keep]
    n = int(keep.sum())
    out = {"n": n, "n_fish_pos": int(y.sum()), "sens": np.nan, "spec": np.nan,
           "ppv": np.nan, "npv": np.nan, "kappa": np.nan, "accuracy": np.nan}
    if n == 0:
        return out
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    out["sens"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["spec"] = tn / (tn + fp) if (tn + fp) else np.nan
    out["ppv"] = tp / (tp + fp) if (tp + fp) else np.nan
    out["npv"] = tn / (tn + fn) if (tn + fn) else np.nan
    out["accuracy"] = (tp + tn) / n
    out["kappa"] = float(cohen_kappa_score(y, p)) if len(set(p)) > 1 and len(set(y)) > 1 else np.nan
    return out


def validate_against_fish(calls_csv: str, fish_csv: str, id_col: str = "patient_id",
                          subtypes: Optional[List[str]] = None) -> pd.DataFrame:
    """Per-subtype sens/spec/PPV/NPV/kappa of the sequencing calls vs real FISH."""
    calls = pd.read_csv(calls_csv).set_index(id_col)
    fish = pd.read_csv(fish_csv).set_index(id_col)
    calls.index = calls.index.astype(str); fish.index = fish.index.astype(str)
    common_pts = calls.index.intersection(fish.index)
    cols = [c for c in (subtypes or fish.columns) if c in calls.columns and c in fish.columns]
    rows = []
    for c in cols:
        m = binary_agreement(fish.loc[common_pts, c].values, calls.loc[common_pts, c].values)
        rows.append({"subtype": c, "gold": "FISH", "is_fish": True, **m})
    return pd.DataFrame(rows)


def run_fish_ready(cfg: dict) -> Optional[pd.DataFrame]:
    """Run iff cfg.paths.fish is set and the file exists; else return None."""
    fish_path = cfg.get("paths", {}).get("fish")
    calls_path = cfg.get("paths", {}).get("cytogenetics", "data/real/cytogenetics.csv")
    if not fish_path or not Path(fish_path).exists():
        return None
    res = validate_against_fish(calls_path, fish_path)
    outdir = Path(cfg.get("paths", {}).get("outdir", "outputs/validation"))
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(outdir / "fish_concordance.csv", index=False)
    return res
