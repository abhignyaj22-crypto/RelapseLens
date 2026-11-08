"""Statistical helpers for endpoint-correct, claim-gated survival evaluation.

All functions are pure (operate on numpy arrays) so they can be unit-tested and
reused across the residual report and any future endpoint.
"""
from __future__ import annotations

import numpy as np

from ..metrics import fast_c_index as harrell_c_index


# --------------------------------------------------------------------------- #
# Paired delta C-index (same test patients) with bootstrap CI
# --------------------------------------------------------------------------- #
def paired_delta_cindex(t, e, risk_a, risk_b, n_boot=2000, seed=42):
    """C(A) - C(B) with a PAIRED bootstrap over the same resampled patients.

    A is the richer model, B the baseline. Returns dict with point estimate,
    95% CI, one-sided bootstrap p (P[delta<=0]), and a claim label.
    """
    t = np.asarray(t, float); e = np.asarray(e, int)
    ra = np.asarray(risk_a, float); rb = np.asarray(risk_b, float)
    c_a = harrell_c_index(t, e, ra)
    c_b = harrell_c_index(t, e, rb)
    delta = c_a - c_b
    rng = np.random.default_rng(seed)
    n = len(t)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if e[idx].sum() < 2:
            continue
        da = harrell_c_index(t[idx], e[idx], ra[idx])
        db = harrell_c_index(t[idx], e[idx], rb[idx])
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(da - db)
    deltas = np.asarray(deltas)
    if deltas.size == 0:
        lo = hi = tail = np.nan
    else:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        tail = float(np.mean(deltas <= 0))  # bootstrap tail mass, NOT a calibrated p-value
    p_perm = paired_permutation_test(t, e, ra, rb, n_perm=n_boot, seed=seed)
    if np.isfinite(lo) and lo > 0:
        claim = "confirmed_improvement"
    elif delta > 0:
        claim = "hypothesis_generating_improvement"
    else:
        claim = "no_evidence_of_improvement"
    return {"cindex_model_a": round(float(c_a), 4), "cindex_model_b": round(float(c_b), 4),
            "delta_cindex": round(float(delta), 4),
            "delta_ci_low": round(float(lo), 4) if np.isfinite(lo) else np.nan,
            "delta_ci_high": round(float(hi), 4) if np.isfinite(hi) else np.nan,
            "p_permutation": round(float(p_perm), 4) if np.isfinite(p_perm) else np.nan,
            "bootstrap_tail_mass": round(float(tail), 4) if np.isfinite(tail) else np.nan,
            "claim": claim}


def paired_delta_ibs(time, event, surv_a, surv_b, grid, n_boot=500, seed=42):
    """Paired-bootstrap Δ integrated-Brier-score, ΔIBS = IBS(B) - IBS(A), on the SAME
    resampled patients and the SAME grid (so both models share one IPCW censoring estimate
    per resample). A is the model expected to be BETTER-calibrated (lower IBS); ΔIBS>0 favours
    A. Returns point estimate, 95% CI, and ci_excludes_zero. Makes the calibration win a
    statistical claim, matching the ΔC-index rigor (not a bare IBS inequality)."""
    from sksurv.metrics import integrated_brier_score
    from sksurv.util import Surv
    time = np.asarray(time, float); event = np.asarray(event, int)
    surv_a = np.asarray(surv_a, float); surv_b = np.asarray(surv_b, float)
    grid = np.asarray(grid, float)

    def _delta(idx):
        y = Surv.from_arrays(event[idx].astype(bool), time[idx].astype(float))
        try:
            ia = integrated_brier_score(y, y, surv_a[idx], grid)
            ib = integrated_brier_score(y, y, surv_b[idx], grid)
            return float(ib - ia)
        except (ValueError, ZeroDivisionError):
            return np.nan

    obs = _delta(np.arange(len(time)))
    rng = np.random.default_rng(seed)
    n = len(time)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if event[idx].sum() < 2:
            continue
        d = _delta(idx)
        if np.isfinite(d):
            deltas.append(d)
    deltas = np.asarray(deltas)
    if deltas.size == 0 or not np.isfinite(obs):
        lo = hi = np.nan
    else:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_ibs": round(float(obs), 5) if np.isfinite(obs) else np.nan,
            "delta_ci_low": round(float(lo), 5) if np.isfinite(lo) else np.nan,
            "delta_ci_high": round(float(hi), 5) if np.isfinite(hi) else np.nan,
            "ci_excludes_zero": bool(np.isfinite(lo) and lo > 0)}


# --------------------------------------------------------------------------- #
# Paired permutation test + multiplicity control (calibrated inference)
# --------------------------------------------------------------------------- #
def paired_permutation_test(t, e, risk_a, risk_b, n_perm=2000, seed=42) -> float:
    """Two-sided paired permutation p-value for ΔC = C(A) - C(B) on the SAME patients.

    Under H0 the two models' risk labels are exchangeable per patient, so we sign-flip
    (swap risk_a <-> risk_b) independently per patient and recompute ΔC. p is the
    add-one-corrected fraction of permutations with |ΔC*| >= |ΔC_obs|. Unlike the
    bootstrap tail mass this is a calibrated test statistic.
    """
    t = np.asarray(t, float); e = np.asarray(e, int)
    ra = np.asarray(risk_a, float); rb = np.asarray(risk_b, float)
    obs = harrell_c_index(t, e, ra) - harrell_c_index(t, e, rb)
    if not np.isfinite(obs):
        return np.nan
    rng = np.random.default_rng(seed)
    n = len(t)
    count = valid = 0
    for _ in range(int(n_perm)):
        flip = rng.random(n) < 0.5
        d = harrell_c_index(t, e, np.where(flip, rb, ra)) - harrell_c_index(t, e, np.where(flip, ra, rb))
        if np.isfinite(d):
            valid += 1
            if abs(d) >= abs(obs) - 1e-12:
                count += 1
    return (count + 1) / (valid + 1) if valid else np.nan


def multiplicity_adjust(pvals, method: str = "bh"):
    """Benjamini-Hochberg (FDR) or Holm (FWER) adjusted p-values. NaNs are ignored in
    the count m and returned as NaN. Returns a list aligned with the input order."""
    p = np.asarray(pvals, float)
    finite = np.where(np.isfinite(p))[0]
    adj = np.full(p.shape, np.nan)
    m = finite.size
    if m == 0:
        return adj.tolist()
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    out = np.empty(m)
    if method == "holm":
        prev = 0.0
        for i, val in enumerate(ranked):
            prev = max(prev, min((m - i) * val, 1.0))
            out[i] = prev
    else:  # Benjamini-Hochberg
        nxt = 1.0
        for i in range(m - 1, -1, -1):
            nxt = min(nxt, min(ranked[i] * m / (i + 1), 1.0))
            out[i] = nxt
    restored = np.empty(m)
    restored[order] = out
    adj[finite] = restored
    return adj.tolist()


# --------------------------------------------------------------------------- #
# Evidence-level labelling
# --------------------------------------------------------------------------- #
def evidence_level(endpoint_type, proposal_target, delta_cindex, delta_ci_low,
                   external_validation_available):
    """Map a result to confirmatory / hypothesis_generating / technical_validation_only
    / unsupported per the agreed rules."""
    relapse_targets = {"relapse", "pfs", "progression", "time_to_progression",
                       "time_to_relapse", "early_progression"}
    if endpoint_type in {"overall_survival", "os"} and proposal_target in relapse_targets:
        return "technical_validation_only"
    if (delta_ci_low is not None and np.isfinite(delta_ci_low)
            and delta_ci_low > 0 and external_validation_available):
        return "confirmatory"
    if delta_cindex is not None and delta_cindex > 0 and (delta_ci_low is None or delta_ci_low <= 0):
        return "hypothesis_generating"
    return "unsupported"


# --------------------------------------------------------------------------- #
# Calibration (needs predicted EVENT probability at a landmark horizon)
# --------------------------------------------------------------------------- #
def calibration_metrics(t, e, p_event, horizon, n_bins=10):
    """Calibration of predicted event probability by `horizon`.

    Observed risk per bin uses 1 - KM(horizon) within the bin (handles censoring).
    Returns scalar metrics + a per-decile table (list of dicts).
    """
    t = np.asarray(t, float); e = np.asarray(e, int); p = np.asarray(p_event, float)
    # Brier score at horizon with IPCW-free approximation (events known by horizon)
    y = ((t <= horizon) & (e == 1)).astype(float)
    known = ~((t < horizon) & (e == 0))  # exclude censored-before-horizon from naive Brier
    brier = float(np.mean((p[known] - y[known]) ** 2)) if known.sum() else np.nan

    # calibration slope (regression of observed KM risk on predicted) + in-the-large
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    rows, obs_all, pred_all = [], [], []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        obs = _km_event_prob(t[idx], e[idx], horizon)
        pred = float(np.mean(p[idx]))
        rows.append({"decile": i + 1, "n": int(len(idx)), "pred_event_prob": round(pred, 4),
                     "observed_event_prob": round(obs, 4) if np.isfinite(obs) else np.nan})
        if np.isfinite(obs):
            obs_all.append(obs); pred_all.append(pred)
    cal_slope = cal_large = np.nan
    if len(obs_all) >= 3:
        pred_all = np.array(pred_all); obs_all = np.array(obs_all)
        A = np.vstack([np.ones_like(pred_all), pred_all]).T
        coef, *_ = np.linalg.lstsq(A, obs_all, rcond=None)
        cal_slope = float(coef[1])                                   # regression slope
        cal_large = float(np.mean(obs_all) - np.mean(pred_all))      # mean obs - mean pred
    return {"brier_at_horizon": round(brier, 4) if np.isfinite(brier) else np.nan,
            "calibration_slope": round(cal_slope, 4) if np.isfinite(cal_slope) else np.nan,
            "calibration_in_the_large": round(cal_large, 4) if np.isfinite(cal_large) else np.nan,
            "horizon": horizon}, rows


def _km_event_prob(t, e, horizon):
    """1 - KM(horizon): observed cumulative event probability with censoring."""
    t = np.asarray(t, float); e = np.asarray(e, int)
    order = np.argsort(t)
    t, e = t[order], e[order]
    surv, n = 1.0, len(t)
    at_risk = n
    seen = set()
    for i in range(n):
        if t[i] > horizon:
            break
        if e[i] == 1:
            surv *= (1 - 1.0 / at_risk)
        at_risk -= 1
        if at_risk <= 0:
            break
    return 1.0 - surv


# --------------------------------------------------------------------------- #
# Decision-curve analysis (net benefit at risk thresholds)
# --------------------------------------------------------------------------- #
def decision_curve(t, e, models: dict[str, np.ndarray], horizon,
                   thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Net benefit at each threshold for each model + treat-all / treat-none.

    models: name -> predicted event probability by horizon. Event prevalence and
    treat-all benefit use KM at horizon to respect censoring.
    """
    t = np.asarray(t, float); e = np.asarray(e, int)
    n = len(t)
    prev = _km_event_prob(t, e, horizon)
    rows = []
    for th in thresholds:
        w = th / (1 - th)
        row = {"threshold": th, "treat_all": round(prev - (1 - prev) * w, 4), "treat_none": 0.0}
        for name, p in models.items():
            p = np.asarray(p, float)
            flagged = p >= th
            nf = flagged.sum()
            if nf == 0:
                row[name] = 0.0
                continue
            tp_rate = _km_event_prob(t[flagged], e[flagged], horizon) * (nf / n)
            fp_rate = (1 - _km_event_prob(t[flagged], e[flagged], horizon)) * (nf / n)
            row[name] = round(float(tp_rate - fp_rate * w), 4)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# NRI / IDI (predicted event probability, two models)
# --------------------------------------------------------------------------- #
def nri_idi(t, e, p_old, p_new, horizon, n_cat_bins=(0.1, 0.2, 0.3)):
    """Categorical NRI + IDI comparing p_old vs p_new at a horizon.

    Events = observed by horizon (excludes censored-before-horizon for NRI/IDI).
    """
    t = np.asarray(t, float); e = np.asarray(e, int)
    p_old = np.asarray(p_old, float); p_new = np.asarray(p_new, float)
    known = ~((t < horizon) & (e == 0))
    y = ((t <= horizon) & (e == 1)).astype(int)[known]
    po, pn = p_old[known], p_new[known]
    cats = np.array(n_cat_bins)

    def cat(p):
        return np.digitize(p, cats)
    co, cn = cat(po), cat(pn)
    ev, nev = y == 1, y == 0
    nri_ev = (np.sum(cn[ev] > co[ev]) - np.sum(cn[ev] < co[ev])) / max(ev.sum(), 1)
    nri_nev = (np.sum(cn[nev] < co[nev]) - np.sum(cn[nev] > co[nev])) / max(nev.sum(), 1)
    nri = nri_ev + nri_nev
    idi = (np.mean(pn[ev]) - np.mean(po[ev])) - (np.mean(pn[nev]) - np.mean(po[nev])) \
        if ev.sum() and nev.sum() else np.nan
    return {"nri": round(float(nri), 4), "nri_events": round(float(nri_ev), 4),
            "nri_nonevents": round(float(nri_nev), 4),
            "idi": round(float(idi), 4) if np.isfinite(idi) else np.nan,
            "n_events": int(ev.sum()), "n_nonevents": int(nev.sum()), "horizon": horizon}
