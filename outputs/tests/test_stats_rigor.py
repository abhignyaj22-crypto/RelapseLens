"""Tests for the calibrated inference layer: paired permutation test + multiplicity."""
import numpy as np

from mm_tte_survival.evaluation.stats import (
    paired_permutation_test, multiplicity_adjust, paired_delta_cindex)


def _data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    t = rng.weibull(1.4, n) * 20 + 1
    e = (rng.random(n) < 0.6).astype(int)
    return t, e, rng


def test_permutation_p_is_calibrated_range():
    t, e, rng = _data()
    r = rng.normal(size=len(t))
    # A == B (identical risk) -> no difference -> large p
    p_same = paired_permutation_test(t, e, r, r.copy())
    assert 0.0 <= p_same <= 1.0 and p_same > 0.3
    # A strongly aligned with outcome, B pure noise -> small p
    risk_signal = -t + rng.normal(0, 0.1, len(t))   # higher risk => shorter time
    p_diff = paired_permutation_test(t, e, risk_signal, rng.normal(size=len(t)))
    assert p_diff < p_same


def test_multiplicity_bh_and_holm_properties():
    pvals = [0.001, 0.01, 0.04, 0.2, 0.5]
    bh = multiplicity_adjust(pvals, "bh")
    holm = multiplicity_adjust(pvals, "holm")
    # adjusted >= raw, bounded in [0,1], Holm at least as conservative as BH
    for raw, b, h in zip(pvals, bh, holm):
        assert raw - 1e-9 <= b <= 1.0
        assert raw - 1e-9 <= h <= 1.0
        assert h >= b - 1e-9
    # BH is monotone non-decreasing in sorted p-order
    assert bh == sorted(bh)


def test_multiplicity_handles_nan():
    out = multiplicity_adjust([0.01, np.nan, 0.02], "bh")
    assert np.isnan(out[1]) and np.isfinite(out[0]) and np.isfinite(out[2])


def test_paired_delta_reports_permutation_not_just_tailmass():
    t, e, rng = _data()
    a = rng.normal(size=len(t)); b = rng.normal(size=len(t))
    out = paired_delta_cindex(t, e, a, b, n_boot=500)
    assert "p_permutation" in out and "bootstrap_tail_mass" in out
    assert "p_bootstrap" not in out            # the misleading label is gone
    assert 0.0 <= out["p_permutation"] <= 1.0
