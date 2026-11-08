"""Conformalized survival LPB — finite-sample marginal coverage guarantee."""
from __future__ import annotations

import numpy as np

from mm_tte_survival.conformal_survival import conformal_lpb, lpb_coverage


def _draw(n, rng, beta=1.0):
    x = rng.normal(size=n)
    rate = np.exp(beta * x)                       # true Cox-exponential hazard
    t = rng.exponential(scale=1.0 / rate)         # true event times, no censoring
    return x, t, np.ones(n, dtype=int)


def test_conformal_lpb_hits_nominal_when_well_specified():
    rng = np.random.default_rng(0)
    xtr, ttr, etr = _draw(3000, rng)
    xcal, tcal, ecal = _draw(3000, rng)
    xte, tte, ete = _draw(4000, rng)
    beta = 1.0
    out = conformal_lpb(beta * xtr, ttr, etr, beta * xcal, tcal, ecal, beta * xte, alpha=0.1)
    cov = lpb_coverage(out["L_conformal"], tte, ete)["coverage"]
    assert 0.86 <= cov <= 0.95, cov          # ~90% target, finite-sample band


def test_conformal_covers_even_when_model_is_misspecified():
    """The conformal guarantee is marginal and model-agnostic: a CONSTANT (X-blind) predictor still
    yields >= 1-alpha marginal coverage after calibration — the property a bare Cox quantile lacks."""
    rng = np.random.default_rng(7)
    xtr, ttr, etr = _draw(3000, rng)
    xcal, tcal, ecal = _draw(3000, rng)
    xte, tte, ete = _draw(4000, rng)
    zero = lambda a: np.zeros_like(a, dtype=float)   # deliberately ignore X
    out = conformal_lpb(zero(ttr), ttr, etr, zero(tcal), tcal, ecal, zero(tte), alpha=0.1)
    cov = lpb_coverage(out["L_conformal"], tte, ete)["coverage"]
    assert cov >= 0.85, cov                  # marginal coverage holds despite misspecification


def test_returns_none_when_too_few_calibration_events():
    rng = np.random.default_rng(1)
    x, t, e = _draw(8, rng)
    assert conformal_lpb(x, t, e, x, t, np.zeros_like(e), x, alpha=0.1) is None
