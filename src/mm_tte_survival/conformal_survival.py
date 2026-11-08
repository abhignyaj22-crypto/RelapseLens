"""Conformalized survival — distribution-free, finite-sample lower predictive bounds on time-to-event.

E3 novel core (autoresearch). Implements a split-conformal lower predictive bound (LPB) in the spirit of
Candès, Lei & Ren (2023, "Conformalized survival analysis", JRSS-B): given a fitted Cox model, calibrate
a survival threshold on a held-out split so that the returned bound L(x) satisfies

    P( T >= L(x) )  >=  1 - alpha        (finite-sample, marginal)

under independent censoring. The point survival predictions are unchanged (so IBS/D-cal are identical to
the underlying Cox) — the wrapper only adds a *calibrated* lower bound the bare model does not provide.

Calibration uses the probability-integral-transform score F̂(T̃|x) on the OBSERVED-EVENT subpopulation of
the calibration split; the conformal alpha-quantile of those scores replaces the nominal alpha, correcting
mis-calibration of the fitted model. Coverage is honestly evaluated only on the determinate set (points
whose status relative to L is known despite censoring).
"""
from __future__ import annotations

import numpy as np

from .survival_curves import breslow_baseline, cox_survival


def calibration_scores(eta_tr, t_tr, e_tr, eta_cal, t_cal, e_cal):
    """PIT conformity scores F̂(T̃|x) on the OBSERVED-EVENT calibration points, at exact observed times."""
    t_tr, e_tr = np.asarray(t_tr, float), np.asarray(e_tr, int)
    eta_tr = np.asarray(eta_tr, float)
    H_cal = breslow_baseline(t_tr, e_tr, eta_tr, np.asarray(t_cal, float))
    F_cal = 1.0 - np.exp(-np.exp(np.asarray(eta_cal, float)) * H_cal)
    return F_cal[np.asarray(e_cal, int) == 1]


def conformal_quantile(scores, alpha, min_events: int = 10):
    """Finite-sample conformal alpha-quantile of PIT scores; None if too few calibration events."""
    scores = np.asarray(scores, float)
    n = int(scores.size)
    if n < min_events:
        return None, n
    k = min(max(int(np.ceil(alpha * (n + 1))), 1), n)
    return float(np.sort(scores)[k - 1]), n


def lpb_at_q(eta_te, t_tr, e_tr, eta_tr, q, grid=None):
    """Lower predictive bound L(x)=inf{t: F̂(t|x) >= q} on the Breslow-support grid; caps at horizon."""
    t_tr, e_tr = np.asarray(t_tr, float), np.asarray(e_tr, int)
    if grid is None:
        ev = np.unique(t_tr[e_tr == 1])
        grid = ev if ev.size >= 2 else None
    if grid is None or len(grid) < 2:
        return None
    H = breslow_baseline(t_tr, e_tr, np.asarray(eta_tr, float), grid)
    F_te = 1.0 - cox_survival(np.asarray(eta_te, float), H)                    # (n_te, G), increasing in t
    mask = F_te >= q
    return np.where(mask.any(axis=1), np.asarray(grid)[mask.argmax(axis=1)], grid[-1])


def conformal_lpb(eta_tr, t_tr, e_tr, eta_cal, t_cal, e_cal, eta_te,
                  alpha: float = 0.1, grid=None):
    """Marginal split-conformal lower predictive bound. eta_* are Cox linear predictors (log-hazard).
    Returns dict with L_conformal, L_naive (uncalibrated nominal-alpha bound), q_hat, grid, n_cal_events,
    or None if the calibration set has too few events to calibrate."""
    scores = calibration_scores(eta_tr, t_tr, e_tr, eta_cal, t_cal, e_cal)
    q_hat, n = conformal_quantile(scores, alpha)
    if q_hat is None:
        return None
    t_tr, e_tr = np.asarray(t_tr, float), np.asarray(e_tr, int)
    if grid is None:
        ev = np.unique(t_tr[e_tr == 1])
        grid = ev if ev.size >= 2 else None
    if grid is None or len(grid) < 2:
        return None
    L_conf = lpb_at_q(eta_te, t_tr, e_tr, eta_tr, q_hat, grid=grid)
    L_naive = lpb_at_q(eta_te, t_tr, e_tr, eta_tr, alpha, grid=grid)
    return {"L_conformal": L_conf, "L_naive": L_naive,
            "q_hat": q_hat, "alpha": float(alpha), "grid": np.asarray(grid), "n_cal_events": n}


def lpb_coverage(L, t_te, e_te):
    """Honest coverage of a lower bound under censoring. Two estimands:
    - coverage_events = P(T >= L | observed event) — the population this PIT-conformal CALIBRATES to;
      the estimand for the guarantee "among patients who die, >=1-alpha die no earlier than L(x)".
    - coverage (determinate) = fraction with T >= L among points whose status vs L is known (covered iff
      T̃ >= L for any censoring status; a violation needs an OBSERVED event before L). Reported as a
      secondary, conservative marginal read; censored-before-L points are indeterminate (excluded)."""
    L = np.asarray(L, float)
    t, e = np.asarray(t_te, float), np.asarray(e_te, int)
    covered = t >= L
    early_event = (e == 1) & (t < L)
    determinate = covered | early_event
    n_det = int(determinate.sum())
    n_ev = int((e == 1).sum())
    return {"coverage_events": float((covered & (e == 1)).sum() / n_ev) if n_ev else float("nan"),
            "n_events": n_ev,
            "coverage": float(covered.sum() / n_det) if n_det else float("nan"),
            "n_determinate": n_det,
            "n_indeterminate": int(((e == 0) & (t < L)).sum()),
            "n": int(t.size)}
