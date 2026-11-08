"""Serial-lab TRAJECTORY features for landmark survival — the *change* a static baseline Cox cannot see.

B2 (run_landmark_os.py) showed LOCF *levels* do not beat a baseline Cox on OS: at an early landmark the
most-recent value ≈ the baseline value, so the dynamic arm had almost no extra information. The genuine
longitudinal signal is rate-of-change — a falling albumin/hemoglobin, a rising LDH/β2M/M-protein. This module
turns a patient's pre-landmark visit history into trend features (slope, delta-from-baseline, variability) that
a baseline-only model structurally cannot represent. Pure functions (no I/O) so they are unit-testable and
leak-proof by construction (the caller passes history already truncated to visits <= the landmark).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VT_COL = "visit_t_days"


def lab_trajectory(hist: pd.DataFrame, lab: str, vt_col: str = VT_COL) -> dict:
    """Per-lab summary from one patient's visits (already truncated to <= the landmark):
    base (earliest), last (LOCF / most-recent), delta (last-base), slope (OLS per day), var (std), nobs.
    Missing/absent lab -> NaN level with 0 change (no trend evidence), never fabricated."""
    if lab in hist.columns:
        m = hist[[vt_col, lab]].dropna()
        vals = m[lab].to_numpy(float)
        days = m[vt_col].to_numpy(float)
    else:
        vals, days = np.array([]), np.array([])
    n = len(vals)
    base = float(vals[0]) if n else np.nan
    last = float(vals[-1]) if n else np.nan
    delta = float(last - base) if n >= 2 else 0.0
    slope = float(np.polyfit(days, vals, 1)[0]) if (n >= 2 and np.ptp(days) > 0) else 0.0  # per day
    var = float(np.std(vals)) if n >= 2 else 0.0
    return {f"base_{lab}": base, f"last_{lab}": last, f"delta_{lab}": delta,
            f"slope_{lab}": slope, f"var_{lab}": var, f"nobs_{lab}": float(n)}


def trajectory_row(hist: pd.DataFrame, labs: list[str], vt_col: str = VT_COL) -> dict:
    """Flatten per-lab trajectory features for one patient's (truncated) history."""
    out: dict = {}
    for lab in labs:
        out.update(lab_trajectory(hist, lab, vt_col))
    return out


def arm_columns(labs: list[str]) -> dict[str, list[str]]:
    """Feature-column sets for the apples-to-apples arms.
    static = baseline level only (the B2 static arm); locf = most-recent level (the B2 dynamic arm);
    trajectory = baseline level PLUS the change features (delta/slope/variability) — isolates whether *change*
    adds discrimination on top of the baseline a static Cox already has."""
    return {
        "static": [f"base_{c}" for c in labs],
        "locf": [f"last_{c}" for c in labs],
        "trajectory": ([f"base_{c}" for c in labs] + [f"delta_{c}" for c in labs]
                       + [f"slope_{c}" for c in labs] + [f"var_{c}" for c in labs]),
    }


# the change-only columns (used by the permutation null: shuffle these across patients -> gain must collapse)
def change_columns(labs: list[str]) -> list[str]:
    return [f"delta_{c}" for c in labs] + [f"slope_{c}" for c in labs] + [f"var_{c}" for c in labs]
