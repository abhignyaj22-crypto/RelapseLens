"""Serial (longitudinal) clinical substrate — Stage 1 clock layer.

Transforms the one-row-per-patient grain into a per-visit table and GATES it: a neural trajectory model
may only be built if a REAL serial substrate with adequate visit density exists. Fabricated/demo data and
baseline-only (one-visit-per-patient) tables fail the gate loudly — no synthetic trajectory is ever
learned. This module owns schema validation, temporal-order validation, and the substrate gate; the
landmark/endpoint construction lives in landmark_builder.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PID, VT, ET, ETYPE = "patient_id", "visit_t_days", "event_t_days", "event_type"
LAB_COLS = ["mprotein_g_dl", "flc_involved_mg_l", "flc_uninvolved_mg_l", "hemoglobin_g_dl",
            "calcium_mg_dl", "creatinine_mg_dl", "albumin_g_dl", "b2m_mg_l", "ldh_u_l"]
REQUIRED_COLS = [PID, VT, ET, ETYPE]


class SubstrateGateError(RuntimeError):
    """Raised when the serial substrate is absent, synthetic, or too sparse to model honestly."""


def build_patient_visit_table(df: pd.DataFrame, *, provenance: dict) -> pd.DataFrame:
    """Validate + normalize a per-visit table. `provenance` must carry {'source': str, 'is_real': bool};
    an is_real=False source is retained ONLY for code tests and is refused by require_real_serial_substrate."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SubstrateGateError(f"serial table missing required columns {missing}")
    v = df.copy()
    v[PID] = v[PID].astype(str)
    for c in [VT, ET] + [x for x in LAB_COLS if x in v.columns]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v = v.sort_values([PID, VT], kind="mergesort").reset_index(drop=True)
    v.attrs["provenance"] = {"source": str(provenance.get("source", "?")),
                             "is_real": bool(provenance.get("is_real", False))}
    validate_temporal_order(v)
    return v


def validate_temporal_order(v: pd.DataFrame) -> None:
    """Fail loud on any temporal inconsistency: negative times, non-monotone visits, or an event that
    precedes the last observed visit (which would make 'future' endpoints leak into the history)."""
    if (v[VT] < 0).any() or (v[ET] < 0).any():
        raise SubstrateGateError("negative visit/event times present")
    for pid, g in v.groupby(PID, sort=False):
        t = g[VT].to_numpy()
        if np.any(np.diff(t) < 0):
            raise SubstrateGateError(f"visits not non-decreasing for patient {pid}")
        et = g[ET].dropna().to_numpy()
        if et.size and float(et[0]) < float(t[-1]) - 1e-9:
            raise SubstrateGateError(
                f"patient {pid}: event_t_days {et[0]} precedes last visit {t[-1]} — endpoint would leak")


@dataclass
class SubstrateReport:
    n_patients: int
    n_visits: int
    median_visits_per_patient: float
    min_visits_per_patient: int
    frac_multi_visit: float
    is_real: bool
    passes: bool
    reasons: list = field(default_factory=list)


def assess_substrate(v: pd.DataFrame, *, min_patients: int = 100, min_median_visits: float = 3.0,
                     min_frac_multi_visit: float = 0.8) -> SubstrateReport:
    counts = v.groupby(PID)[VT].size()
    is_real = bool(v.attrs.get("provenance", {}).get("is_real", False))
    reasons = []
    if len(counts) < min_patients:
        reasons.append(f"too few patients ({len(counts)} < {min_patients})")
    med = float(counts.median()) if len(counts) else 0.0
    if med < min_median_visits:
        reasons.append(f"median visits/patient {med} < {min_median_visits} (baseline-only?)")
    frac_multi = float((counts >= 2).mean()) if len(counts) else 0.0
    if frac_multi < min_frac_multi_visit:
        reasons.append(f"only {frac_multi:.2f} of patients have >=2 visits (< {min_frac_multi_visit})")
    if not is_real:
        reasons.append("provenance is_real=False (synthetic/demo data cannot ground a trajectory claim)")
    return SubstrateReport(n_patients=len(counts), n_visits=int(len(v)),
                           median_visits_per_patient=med, min_visits_per_patient=int(counts.min()) if len(counts) else 0,
                           frac_multi_visit=frac_multi, is_real=is_real,
                           passes=(not reasons), reasons=reasons)


def require_real_serial_substrate(v: pd.DataFrame, **kw) -> SubstrateReport:
    """STAGE-1 ACCEPTANCE GATE. Returns the report if the substrate is real + dense enough; otherwise
    raises SubstrateGateError. Callers building a neural trajectory model MUST pass through this — a
    failing gate means STOP (do not train a trajectory model on absent/synthetic/sparse data)."""
    rep = assess_substrate(v, **kw)
    if not rep.passes:
        raise SubstrateGateError("serial substrate FAILED Stage-1 gate: " + "; ".join(rep.reasons)
                                 + " -> do NOT build a neural trajectory model on this data.")
    return rep
