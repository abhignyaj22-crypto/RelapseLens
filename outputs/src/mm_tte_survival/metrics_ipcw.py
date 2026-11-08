"""IPCW (censoring-distribution-aware) discrimination + calibration metrics.

Harrell's C (see metrics.py) is censoring-distribution dependent and upward-biased
at the small event counts in this cohort (n_events ~ 38-182). These are the PRIMARY
leaderboard metrics:

  * uno_c_index            — Uno's IPCW concordance (weights by the censoring dist)
  * cumulative_dynamic_auc — time-dependent AUC(t), returned as its mean over a grid
  * d_calibration          — Haider distribution-calibration chi-square p-value

The IPCW integrated Brier score lives in survival_curves.ipcw_ibs (reused, not
duplicated). All functions degrade to NaN (never raise) on degenerate inputs so a
single unstable fold cannot abort a whole leaderboard.
"""
from __future__ import annotations

import numpy as np
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc

from .survival_curves import time_grid


def _surv(e, t):
    return Surv.from_arrays(np.asarray(e).astype(bool), np.asarray(t, dtype=float))


def uno_c_index(t_train, e_train, t_test, e_test, risk_test, tau=None) -> float:
    """Uno's IPCW concordance. `risk_test`: higher = higher risk (e.g. Cox log partial
    hazard). `tau` defaults to a truncation strictly inside the followed-up support of
    both train and test (required by the IPCW weights). NaN if undefined."""
    t_train = np.asarray(t_train, float); t_test = np.asarray(t_test, float)
    e_test = np.asarray(e_test, int)
    if e_test.sum() < 2 or np.asarray(e_train, int).sum() < 2:
        return float("nan")
    if tau is None:
        tau = min(float(t_train.max()), float(t_test.max())) - 1e-6
    try:
        c = concordance_index_ipcw(_surv(e_train, t_train), _surv(e_test, t_test),
                                   np.asarray(risk_test, float), tau=tau)[0]
        return float(c)
    except (ValueError, ZeroDivisionError):
        return float("nan")


def cumulative_dynamic_auc_mean(t_train, e_train, t_test, e_test, risk_test, times=None) -> float:
    """Mean time-dependent (cumulative/dynamic) AUC over a grid inside follow-up.
    Higher risk => earlier event. NaN if no valid grid / degenerate risk-sets.

    Test durations beyond the train censoring support are administratively censored at
    the support cap so the IPCW censoring estimate G(t) (fit on train) is defined
    everywhere it is evaluated — the standard fix for test follow-up exceeding train.
    """
    t_train = np.asarray(t_train, float); t_test = np.asarray(t_test, float)
    e_train = np.asarray(e_train, int); e_test = np.asarray(e_test, int)
    if e_train.sum() < 2 or e_test.sum() < 2:
        return float("nan")
    cap = float(t_train.max()) - 1e-3
    tt = np.minimum(t_test, cap)
    ee = np.where(t_test > cap, 0, e_test)
    if times is None:
        times = time_grid(t_train, e_train, tt, n=8)
    if times is None:
        return float("nan")
    times = np.asarray(times, float)
    times = times[times < cap]
    for _ in range(3):
        if times.size == 0:
            return float("nan")
        try:
            _, mean_auc = cumulative_dynamic_auc(
                _surv(e_train, t_train), _surv(ee, tt), np.asarray(risk_test, float), times)
            return float(mean_auc)
        except (ValueError, ZeroDivisionError):
            times = times[:-1]   # drop a boundary time sksurv rejects, retry
    return float("nan")


def d_calibration(surv_at_obs, event, n_bins: int = 10) -> dict:
    """Haider et al. D-calibration on predicted survival probabilities S_i(t_i) at each
    subject's OBSERVED time. Under a well-calibrated model, S_i(t_i) for events is
    Uniform(0,1); censored subjects contribute their mass spread over [0, S_i(t_i)].
    Returns the per-bin counts, chi-square statistic and p-value (large p = calibrated).
    """
    s = np.clip(np.asarray(surv_at_obs, float), 1e-8, 1.0)
    e = np.asarray(event, int)
    n = len(s)
    if n == 0:
        return {"d_cal_chi2": float("nan"), "d_cal_p": float("nan"), "n_bins": n_bins}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, float)
    # events: a point mass in the bin containing S_i(t_i)
    for si in s[e == 1]:
        counts[min(int(si * n_bins), n_bins - 1)] += 1.0
    # censored: uniform mass over [0, S_i], the portion of each bin below S_i
    for si in s[e == 0]:
        b = min(int(si * n_bins), n_bins - 1)
        counts[b] += (si - edges[b]) / (si) if si > 0 else 0.0
        counts[:b] += (1.0 / n_bins) / si if si > 0 else 0.0
    expected = n / n_bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected)) if expected > 0 else float("nan")
    dof = n_bins - 1
    p = _chi2_sf(chi2, dof) if np.isfinite(chi2) else float("nan")
    return {"d_cal_chi2": round(chi2, 4) if np.isfinite(chi2) else float("nan"),
            "d_cal_p": round(p, 4) if np.isfinite(p) else float("nan"), "n_bins": n_bins}


def _chi2_sf(x: float, dof: int) -> float:
    """Upper-tail chi-square probability via the regularized upper incomplete gamma
    Q(dof/2, x/2), computed with a numerically stable continued fraction / series.
    Avoids a scipy dependency (kept consistent with the dependency-light metrics.py)."""
    if x <= 0:
        return 1.0
    a, xx = dof / 2.0, x / 2.0
    from math import lgamma, log, exp
    if xx < a + 1.0:                      # series for lower incomplete gamma P
        term = 1.0 / a
        summ = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= xx / n
            summ += term
            if abs(term) < abs(summ) * 1e-12:
                break
        P = summ * exp(-xx + a * log(xx) - lgamma(a))
        return max(0.0, min(1.0, 1.0 - P))
    b = xx + 1.0 - a                      # continued fraction for upper incomplete Q
    c = 1e300; d = 1.0 / b; h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    Q = exp(-xx + a * log(xx) - lgamma(a)) * h
    return max(0.0, min(1.0, Q))
