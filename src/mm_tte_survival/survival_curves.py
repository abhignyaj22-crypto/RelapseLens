"""Survival-function utilities: Breslow baseline, S(tau|x), and IPCW-IBS.

These replace the misleading `integrated_brier_proxy` (a sigmoid of standardized
risk, NOT a Brier score). IPCW integrated Brier score is computed with
scikit-survival, which weights by the estimated censoring distribution.
"""
from __future__ import annotations

import numpy as np
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score


def time_grid(t_train, e_train, t_test, n: int = 12):
    """A grid strictly inside the followed-up support of BOTH train and test
    (sksurv requires IBS times below the max observed time in each)."""
    t_train, e_train, t_test = map(np.asarray, (t_train, e_train, t_test))
    ev = t_train[e_train.astype(bool)]
    if ev.size == 0:
        return None
    lo = max(np.quantile(ev, 0.1), float(t_test.min()) + 1e-3, 1.0)
    hi = min(float(t_train.max()), float(t_test.max())) - 1e-3
    if hi <= lo:
        return None
    return np.linspace(lo, hi, n)


def breslow_baseline(t_train, e_train, eta_train, grid):
    """Breslow cumulative-baseline-hazard H0(tau) from a Cox linear predictor."""
    t_train, e_train, eta_train, grid = map(np.asarray, (t_train, e_train, eta_train, grid))
    order = np.argsort(t_train)
    t_s, e_s, r_s = t_train[order], e_train[order], np.exp(eta_train[order])
    rev_cum = np.cumsum(r_s[::-1])[::-1]              # risk-set sum at each time
    inc = np.zeros_like(t_s, dtype=float)
    inc[e_s == 1] = 1.0 / np.clip(rev_cum[e_s == 1], 1e-8, None)
    H_at = np.cumsum(inc)
    return np.array([H_at[t_s <= tau][-1] if np.any(t_s <= tau) else 0.0 for tau in grid])


def cox_survival(eta, H_grid):
    """S(tau|x) = exp(-H0(tau) * exp(eta)), shape (n, len(grid))."""
    return np.exp(-np.outer(np.exp(np.asarray(eta)), np.asarray(H_grid)))


def ipcw_ibs(t_train, e_train, t_test, e_test, S_test, grid) -> float:
    """IPCW integrated Brier score (lower is better). S_test: (n_test, len(grid)).

    Robust to small/degenerate groups: grid times are clamped strictly inside the
    censoring support of BOTH train and test, and any boundary time that sksurv
    still rejects is dropped before retrying. Returns nan if no valid grid remains.
    """
    t_train = np.asarray(t_train, dtype=float); t_test = np.asarray(t_test, dtype=float)
    surv_train = Surv.from_arrays(np.asarray(e_train).astype(bool), t_train)
    surv_test = Surv.from_arrays(np.asarray(e_test).astype(bool), t_test)
    g = np.asarray(grid, dtype=float); S = np.asarray(S_test)
    hi = min(t_train.max(), t_test.max())
    keep = g < hi
    g, S = g[keep], S[:, keep]
    for _ in range(3):
        if g.size == 0:
            return float("nan")
        try:
            return float(integrated_brier_score(surv_train, surv_test, S, g))
        except ValueError:
            g, S = g[:-1], S[:, :-1]   # drop the boundary point sksurv rejects, retry
    return float("nan")
