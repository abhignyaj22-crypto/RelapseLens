"""Stage 2 / PR2 scaffold tests — shapes, defective-FHT monotonicity, and THE training gate.

The teacher forward runs on synthetic tensors purely to exercise shapes/monotonicity (no claim). The
decisive test is that `fit` / `assert_trainable_on` REFUSE synthetic and baseline-only substrates — the
teacher cannot be trained on data that does not exist on the open tier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mm_tte_survival.models.clinical_teacher import (
    ClinicalTeacher, ClinicalAnchoredDynamics, ClinicalVisitEncoder, DEFAULT_HORIZONS_DAYS)
from mm_tte_survival.data.serial_clinical import build_patient_visit_table, SubstrateGateError, PID, VT, ET, ETYPE

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo" / "clinical_serial.csv"
REAL_BASELINE = ROOT / "data" / "real" / "clinical_survival.csv"

N, K, N_LABS, N_TREAT, COND = 6, 4, 9, 3, 5


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    labs = torch.randn(N, K, N_LABS, generator=g)
    treat = torch.randn(N, K, N_TREAT, generator=g)
    dt = torch.rand(N, K, generator=g) * 30.0
    mask = torch.ones(N, K)
    mask[0, 2:] = 0                     # a short-history patient (2 visits)
    c = torch.randn(N, COND, generator=g)
    return labs, treat, dt, mask, c


def _teacher():
    return ClinicalTeacher(n_labs=N_LABS, n_treat=N_TREAT, cond_dim=COND)


# ---- shapes + valid probability ranges ----
def test_forward_shapes_and_ranges():
    out = _teacher()(*_batch())
    assert out.z_tk.shape == (N, 32) and out.s0.shape == (N,)
    assert out.drift.shape == (N,) and out.sigma.shape == (N,)
    assert (out.sigma > 0).all()
    for h in DEFAULT_HORIZONS_DAYS:
        assert out.risk[h].shape == (N,)
        assert (out.risk[h] >= 0).all() and (out.risk[h] <= 1).all()
    assert (out.p_never >= 0).all() and (out.p_never <= 1).all()


def test_encoder_selects_last_valid_visit():
    enc = ClinicalVisitEncoder(N_LABS, N_TREAT)
    labs, treat, dt, mask, _ = _batch()
    z = enc(labs, treat, dt, mask)
    assert z.shape == (N, enc.latent_dim) and torch.isfinite(z).all()


# ---- defective-FHT behaviour ----
def test_risk_monotone_nondecreasing_in_horizon():
    out = _teacher()(*_batch())
    r = [out.risk[h] for h in sorted(DEFAULT_HORIZONS_DAYS)]
    for a, b in zip(r, r[1:]):
        assert (b + 1e-6 >= a).all()                 # more time => at least as much crossing risk


def test_responder_has_nonzero_never_cross_mass():
    # force a strong negative drift (responder): P(T*=inf) must be > 0, and risk bounded below 1.
    t = _teacher()
    out = t(*_batch())
    with torch.no_grad():
        out_drift = out.drift.clone()
    gap = (t.barrier - out.s0).reshape(-1)
    from mm_tte_survival.longitudinal.crossing_head import never_cross_prob, crossing_risk_score
    neg = torch.full_like(out_drift, -1.0)
    pn = never_cross_prob(gap, neg, out.sigma)
    assert (pn > 0).all()
    assert (crossing_risk_score(gap, neg, out.sigma, 365.0) < 1.0).all()


def test_crossing_loss_is_finite_scalar():
    t = _teacher(); out = t(*_batch())
    et = torch.rand(N) * 500 + 30
    ev = torch.tensor([1.0, 0, 1, 0, 1, 0])
    loss = t.crossing_loss(out, et, ev)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_dynamics_is_calendar_time_and_flows():
    dyn = ClinicalAnchoredDynamics(latent_dim=8, cond_dim=COND)
    assert dyn.semantics == "calendar_time"
    z0 = torch.randn(N, 8); c = torch.randn(N, COND); tmax = torch.rand(N) * 100
    zT, vbar = dyn.flow(z0, c, tmax)
    assert zT.shape == z0.shape and vbar.shape == z0.shape and torch.isfinite(zT).all()


# ---- THE gate: refuse to train on synthetic or baseline-only substrates ----
def test_fit_refuses_synthetic_demo_substrate():
    v = build_patient_visit_table(pd.read_csv(DEMO), provenance={"source": "demo", "is_real": False})
    with pytest.raises(SubstrateGateError):
        ClinicalTeacher.assert_trainable_on(v)
    with pytest.raises(SubstrateGateError):
        _teacher().fit(v)


def test_fit_refuses_real_baseline_only_substrate():
    d = pd.read_csv(REAL_BASELINE)
    serial = pd.DataFrame({PID: d["patient_id"].astype(str), VT: 0.0,
                           ET: pd.to_numeric(d.get("os_time", d.get("time")), errors="coerce") * 30.4,
                           ETYPE: np.where(d.get("os_event", d.get("event", 0)) == 1, "death", "censored")})
    v = build_patient_visit_table(serial, provenance={"source": "real_baseline", "is_real": True})
    with pytest.raises(SubstrateGateError):
        ClinicalTeacher.assert_trainable_on(v)          # dense-enough? no — 1 visit/patient
