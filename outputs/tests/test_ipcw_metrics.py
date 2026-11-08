"""Unit tests for the IPCW discrimination/calibration metrics (metrics_ipcw)."""
import numpy as np

from mm_tte_survival.metrics_ipcw import (
    uno_c_index, cumulative_dynamic_auc_mean, d_calibration, _chi2_sf)


def _cohort(seed=0, n=240):
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    t = np.exp(-0.8 * risk) * rng.weibull(1.4, n) * 20 + 1   # higher risk -> earlier event
    e = (rng.random(n) < 0.6).astype(int)
    tr, te = slice(0, n // 2), slice(n // 2, None)
    return (t[tr], e[tr], t[te], e[te], risk[te])


def test_uno_c_index_orientation():
    t_tr, e_tr, t_te, e_te, r = _cohort()
    aligned = uno_c_index(t_tr, e_tr, t_te, e_te, r)
    reversed_ = uno_c_index(t_tr, e_tr, t_te, e_te, -r)
    assert 0.6 < aligned <= 1.0
    assert reversed_ < 0.4
    assert abs((aligned - 0.5) + (reversed_ - 0.5)) < 1e-6   # symmetric around 0.5


def test_uno_c_index_degenerate_is_nan():
    t_tr, e_tr, t_te, _, r = _cohort()
    assert np.isnan(uno_c_index(t_tr, e_tr, t_te, np.zeros_like(r, int), r))  # no test events


def test_td_auc_handles_test_beyond_train_support():
    # a handful of test durations exceed train max; must still return a finite AUC
    t_tr, e_tr, t_te, e_te, r = _cohort()
    t_te = t_te.copy(); t_te[:5] = t_tr.max() * 2.0   # partial beyond-support exceedance
    auc = cumulative_dynamic_auc_mean(t_tr, e_tr, t_te, e_te, r)
    assert np.isfinite(auc) and 0.5 < auc <= 1.0


def test_d_calibration_uniform_is_calibrated():
    rng = np.random.default_rng(1)
    s = rng.random(400)               # uniform S(t_i) for events == well calibrated
    out = d_calibration(s, np.ones(400, int))
    assert out["d_cal_p"] > 0.05

    s_bad = rng.random(400) ** 4      # heavily skewed -> miscalibrated
    assert d_calibration(s_bad, np.ones(400, int))["d_cal_p"] < 0.05


def test_chi2_sf_matches_known_critical_values():
    assert abs(_chi2_sf(0.0, 9) - 1.0) < 1e-9
    assert abs(_chi2_sf(16.919, 9) - 0.05) < 1e-3     # chi2_{0.05, 9}
    assert _chi2_sf(100.0, 9) < 1e-10
