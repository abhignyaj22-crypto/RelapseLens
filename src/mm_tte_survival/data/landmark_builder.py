"""Landmark-example construction — Stage 1 endpoint layer (leak-proof by construction).

A landmark example = (patient history truncated at an index time)  ->  (future endpoint within a horizon).
Every guard the Stage-1 acceptance gate requires is enforced here:
  * history is strictly <= index (no future lab leakage),
  * the endpoint is derived only from information strictly AFTER the index (no post-index leakage),
  * treatment context is as-of-index (treatment_timeline.py; no future-treatment leakage),
  * a latent time axis (forbid_calendar_trajectory_claim) may NOT be given a calendar horizon,
  * landmark examples carry patient_id so splits stay patient-disjoint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .serial_clinical import PID, VT, ET, ETYPE, SubstrateGateError, validate_temporal_order

# event_type tokens that terminate follow-up (positive endpoint) vs administrative censoring
_EVENT_TOKENS = {"progression", "relapse", "death", "next_treatment", "event"}
_CENSOR_TOKENS = {"censored", "censor", "none", ""}


def truncate_history_at_index(v: pd.DataFrame, patient_id: str, index_t: float) -> pd.DataFrame:
    """Return only this patient's visits at or before index_t — the ONLY information a landmark model
    may see. Strictly past-only: any visit after index_t is dropped (no future-lab leakage)."""
    g = v[(v[PID].astype(str) == str(patient_id)) & (v[VT] <= index_t + 1e-9)]
    return g.sort_values(VT, kind="mergesort").reset_index(drop=True)


def derive_endpoint(event_t: float, event_type: str, index_t: float, horizon_days: float) -> dict:
    """Endpoint within (index_t, index_t + horizon]. label=1 iff a terminating event occurs strictly
    after the index and within the horizon; else 0 (event-free through the horizon or censored). The
    event MUST be strictly after the index — an at/pre-index event cannot be a 'future' endpoint."""
    et = float(event_t)
    is_event = str(event_type).strip().lower() in _EVENT_TOKENS
    dt = et - index_t
    if dt <= 0:
        return {"evaluable": False, "reason": "event/censor not strictly after index"}
    horizon_end = index_t + horizon_days
    if is_event and et <= horizon_end + 1e-9:
        return {"evaluable": True, "label": 1, "time_from_index": dt, "censored": False}
    if et >= horizon_end - 1e-9:                       # observed (event or censor) reaches the horizon
        return {"evaluable": True, "label": 0, "time_from_index": float(horizon_days), "censored": not is_event}
    # censored before the horizon -> outcome within horizon unknown
    return {"evaluable": False, "reason": "censored before horizon (indeterminate)"}


def build_landmark_examples(v: pd.DataFrame, *, horizon_days: float, time_axis: dict,
                            index_times: dict | None = None, min_history_visits: int = 1) -> pd.DataFrame:
    """Build leak-proof landmark examples. `time_axis` is a resolved entry from time_axes.yaml.
    `index_times`: optional {patient_id: index_t}; default = each patient's every visit time (visit-anchored).
    Returns one row per (patient, index) with n_history_visits, last-observed labs, endpoint label/time."""
    if time_axis.get("forbid_calendar_trajectory_claim") or time_axis.get("licenses_calendar_time") is False:
        # a latent/pseudotime axis cannot license a calendar-days horizon
        if time_axis.get("semantics") in ("latent", "pseudotime") or time_axis.get("clock_kind") == "pseudotime":
            raise SubstrateGateError(
                "requested a calendar-days horizon on a latent/pseudotime axis "
                "(forbid_calendar_trajectory_claim) — this would fabricate a calendar-time trajectory claim")
    validate_temporal_order(v)
    rows = []
    for pid, g in v.groupby(PID, sort=False):
        g = g.sort_values(VT, kind="mergesort")
        et = g[ET].dropna()
        ev_t = float(et.iloc[0]) if len(et) else np.nan
        ev_type = str(g[ETYPE].iloc[0]) if ETYPE in g else "censored"
        anchors = ([index_times[str(pid)]] if index_times and str(pid) in index_times
                   else g[VT].tolist())
        for idx_t in anchors:
            hist = g[g[VT] <= idx_t + 1e-9]
            if len(hist) < min_history_visits:
                continue
            if not np.isfinite(ev_t):
                continue
            ep = derive_endpoint(ev_t, ev_type, float(idx_t), horizon_days)
            if not ep["evaluable"]:
                continue
            last = hist.iloc[-1]
            rows.append({PID: str(pid), "index_t_days": float(idx_t),
                         "n_history_visits": int(len(hist)),
                         "horizon_days": float(horizon_days),
                         "label": int(ep["label"]), "time_from_index": float(ep["time_from_index"]),
                         "censored": bool(ep["censored"]),
                         **{f"last_{c}": float(last[c]) for c in v.columns
                            if c not in (PID, VT, ET, ETYPE) and pd.api.types.is_numeric_dtype(v[c])}})
    out = pd.DataFrame(rows)
    if len(out):
        _assert_no_leakage(v, out)
    return out


def _assert_no_leakage(v: pd.DataFrame, examples: pd.DataFrame) -> None:
    """Post-hoc guard: no landmark's history could contain a visit after its index (belt-and-suspenders)."""
    vt_by_pid = {str(p): g[VT].to_numpy() for p, g in v.groupby(PID, sort=False)}
    for _, r in examples.iterrows():
        vt = vt_by_pid[str(r[PID])]
        n_le = int(np.sum(vt <= r["index_t_days"] + 1e-9))
        if n_le != int(r["n_history_visits"]):
            raise SubstrateGateError(f"leakage: patient {r[PID]} history count {r['n_history_visits']} "
                                     f"!= visits<=index {n_le}")


def landmark_patient_disjoint_split(examples: pd.DataFrame, seed: int, test_frac: float = 0.3):
    """Split landmark examples so ALL landmarks of a patient share a fold (patient-disjoint)."""
    pids = np.array(sorted(examples[PID].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_test = int(round(test_frac * len(pids)))
    test_pids = set(pids[:n_test].tolist())
    is_test = examples[PID].astype(str).isin(test_pids).to_numpy()
    assert not (set(examples[PID][is_test]) & set(examples[PID][~is_test])), "patient split not disjoint"
    return ~is_test, is_test
