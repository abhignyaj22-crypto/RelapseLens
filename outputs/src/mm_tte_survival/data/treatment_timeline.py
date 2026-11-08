"""Treatment timeline — as-of-index exposure, no future-treatment leakage (Stage 1).

Derives per-patient regimen/line intervals from the serial table and exposes ONLY the treatment context
known at a landmark index (regimens/lines that started at or before the index). A regimen that starts
after the index is invisible to the model — exactly the no-future-treatment-leakage guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .serial_clinical import PID, VT

REG, LINE = "regimen", "line_of_therapy"


def build_treatment_timeline(v: pd.DataFrame) -> pd.DataFrame:
    """One row per (patient, regimen-episode) with the visit time it first appears (start_t_days).
    Episodes are contiguous runs of the same (line, regimen) in visit order — a pragmatic start-time
    proxy when explicit start/stop dates are absent."""
    if REG not in v.columns and LINE not in v.columns:
        return pd.DataFrame(columns=[PID, LINE, REG, "start_t_days"])
    rows = []
    for pid, g in v.groupby(PID, sort=False):
        g = g.sort_values(VT, kind="mergesort")
        prev = None
        for _, r in g.iterrows():
            key = (r.get(LINE, np.nan), str(r.get(REG, "")))
            if key != prev:
                rows.append({PID: str(pid), LINE: r.get(LINE, np.nan),
                             REG: str(r.get(REG, "")), "start_t_days": float(r[VT])})
                prev = key
    return pd.DataFrame(rows)


def treatment_exposure_at(timeline: pd.DataFrame, patient_id: str, index_t: float) -> dict:
    """Treatment context as of index_t: the latest regimen/line that STARTED at or before the index,
    plus days-on-current and number of prior lines. Regimens starting after index_t are excluded."""
    g = timeline[(timeline[PID].astype(str) == str(patient_id)) &
                 (timeline["start_t_days"] <= index_t + 1e-9)]
    if not len(g):
        return {"current_regimen": None, "current_line": np.nan, "days_on_current": np.nan,
                "n_prior_lines": 0}
    g = g.sort_values("start_t_days", kind="mergesort")
    last = g.iloc[-1]
    lines = pd.to_numeric(g[LINE], errors="coerce").dropna().unique()
    return {"current_regimen": last[REG], "current_line": float(last[LINE]) if pd.notna(last[LINE]) else np.nan,
            "days_on_current": float(index_t - last["start_t_days"]),
            "n_prior_lines": int(max(len(lines) - 1, 0))}
