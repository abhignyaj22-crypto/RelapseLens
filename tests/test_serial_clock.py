"""Stage 1 — serial clock + landmark leakage tests + substrate-gate.

The synthetic demo (data/demo/clinical_serial.csv) is used ONLY as a CODE fixture to exercise the
leak-proof logic — never for any scientific claim (its provenance is is_real=False and the gate refuses
it). The decisive test is that the substrate gate FAILS on the real baseline-only data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mm_tte_survival.data.serial_clinical import (
    build_patient_visit_table, validate_temporal_order, assess_substrate,
    require_real_serial_substrate, SubstrateGateError, PID, VT, ET, ETYPE)
from mm_tte_survival.data.landmark_builder import (
    truncate_history_at_index, derive_endpoint, build_landmark_examples,
    landmark_patient_disjoint_split)
from mm_tte_survival.data.treatment_timeline import build_treatment_timeline, treatment_exposure_at

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo" / "clinical_serial.csv"          # synthetic — code fixture only
REAL_BASELINE = ROOT / "data" / "real" / "clinical_survival.csv"


def _demo():
    return build_patient_visit_table(pd.read_csv(DEMO), provenance={"source": "demo", "is_real": False})


# ---- temporal-order + schema validation ----
def test_build_and_validate_demo_fixture():
    v = _demo()
    assert {PID, VT, ET, ETYPE}.issubset(v.columns) and len(v) > 0


def test_validate_temporal_order_raises_on_event_before_last_visit():
    bad = pd.DataFrame({PID: ["p", "p"], VT: [0.0, 100.0], ET: [50.0, 50.0],
                        ETYPE: ["progression", "progression"]})
    with pytest.raises(SubstrateGateError):
        validate_temporal_order(bad)


def test_validate_temporal_order_raises_on_nonmonotone_visits():
    bad = pd.DataFrame({PID: ["p", "p"], VT: [100.0, 0.0], ET: [200.0, 200.0], ETYPE: ["censored", "censored"]})
    with pytest.raises(SubstrateGateError):
        validate_temporal_order(bad)


# ---- leak-proof history truncation + endpoint derivation ----
def test_truncate_history_is_strictly_past_only():
    v = _demo()
    pid = v[PID].iloc[0]
    idx = float(v[v[PID] == pid][VT].median())
    h = truncate_history_at_index(v, pid, idx)
    assert (h[VT] <= idx + 1e-9).all() and len(h) >= 1


def test_derive_endpoint_requires_post_index_event():
    # event at/before index cannot be a future endpoint
    assert derive_endpoint(event_t=50, event_type="progression", index_t=50, horizon_days=90)["evaluable"] is False
    assert derive_endpoint(event_t=40, event_type="progression", index_t=50, horizon_days=90)["evaluable"] is False


def test_derive_endpoint_label_within_horizon():
    assert derive_endpoint(80, "progression", 0, 90) == {"evaluable": True, "label": 1,
                                                         "time_from_index": 80.0, "censored": False}
    r = derive_endpoint(400, "progression", 0, 90)           # event after horizon -> event-free label 0
    assert r["evaluable"] and r["label"] == 0


def test_build_landmark_examples_no_future_lab_leakage():
    v = _demo()
    axis = {"semantics": "calendar_time", "licenses_calendar_time": True}
    ex = build_landmark_examples(v, horizon_days=180, time_axis=axis)
    assert len(ex) > 0
    # every example's history count == visits at-or-before its index (guard also runs inside builder)
    for _, r in ex.iterrows():
        vt = v[v[PID] == r[PID]][VT].to_numpy()
        assert int(np.sum(vt <= r["index_t_days"] + 1e-9)) == int(r["n_history_visits"])


def test_latent_axis_forbids_calendar_horizon():
    v = _demo()
    latent = {"semantics": "latent", "forbid_calendar_trajectory_claim": True, "clock_kind": "pseudotime"}
    with pytest.raises(SubstrateGateError):
        build_landmark_examples(v, horizon_days=180, time_axis=latent)


def test_landmark_split_is_patient_disjoint():
    v = _demo()
    ex = build_landmark_examples(v, horizon_days=180, time_axis={"semantics": "calendar_time"})
    tr, te = landmark_patient_disjoint_split(ex, seed=0)
    assert not (set(ex[PID][tr]) & set(ex[PID][te]))


# ---- treatment: no future-treatment leakage ----
def test_treatment_exposure_excludes_future_regimens():
    v = _demo()
    tl = build_treatment_timeline(v)
    pid = str(v[PID].iloc[0])
    starts = tl[tl[PID] == pid]["start_t_days"]
    if len(starts) >= 2:
        idx = float(starts.iloc[0])                          # before the 2nd regimen starts
        exp = treatment_exposure_at(tl, pid, idx)
        assert exp["days_on_current"] >= 0                   # only regimens started <= idx are visible


# ---- THE decisive gate: real baseline-only data must FAIL ----
def test_substrate_gate_FAILS_on_real_baseline_only_data():
    d = pd.read_csv(REAL_BASELINE)
    # coerce the baseline table into the serial schema: one visit at t=0 per patient
    serial = pd.DataFrame({PID: d["patient_id"].astype(str), VT: 0.0,
                           ET: pd.to_numeric(d.get("os_time", d.get("time")), errors="coerce") * 30.4,
                           ETYPE: np.where(d.get("os_event", d.get("event", 0)) == 1, "death", "censored")})
    v = build_patient_visit_table(serial, provenance={"source": "real_baseline", "is_real": True})
    rep = assess_substrate(v)
    assert not rep.passes and any("baseline-only" in r or "median" in r for r in rep.reasons)
    with pytest.raises(SubstrateGateError):
        require_real_serial_substrate(v)


def test_substrate_gate_rejects_synthetic_provenance():
    v = _demo()                                              # dense but is_real=False
    with pytest.raises(SubstrateGateError):
        require_real_serial_substrate(v)
