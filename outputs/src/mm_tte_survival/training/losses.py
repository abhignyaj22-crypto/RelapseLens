"""Corrected, numerically-stable, reduction-consistent survival losses for MMTTE.

Drop-in replacement for src/mm_tte_survival/training/losses.py. Public names are
kept backward-compatible with trainer_legacy.py; internals are fixed and new
helpers (survival_at / event_prob_at / survival_curve_distillation) are added so
the layers above can use REAL predicted probabilities instead of a sigmoid-of-risk
hack. Every formula here is verified against scipy references in test_losses.py.

Fixes vs the original (all proven numerically — see REFACTOR_MMTTE_BOTTOM_UP.md):
  * AFT/FHT log-survival via torch.special.log_ndtr (the original log(1-Phi(z))
    saturates at log(eps) for z>~6 -> flat gradients for long survivors).
  * FHT survival via log_ndtr + log1mexp (the original exp(2*lam/mu) overflows to
    ~1e86 then nan_to_num silently replaced NaN grads with a constant).
  * soft_distillation_loss: removed the temperature arg (it was a no-op — dividing
    by T then z-standardizing cancels T exactly).
  * Consistent, explicit `reduction`; documented Breslow-ties caveat for Cox.
  * survival_at(): closed-form S(horizon) for AFT/FHT (a real probability).
"""
from __future__ import annotations
import math
import torch
from torch import Tensor
from torch.special import log_ndtr  # log Phi(x), numerically stable

_LOG_2PI = math.log(2.0 * math.pi)


def _log1mexp(u: Tensor) -> Tensor:
    """Stable log(1 - exp(u)) for u <= 0 (Maechler 2012)."""
    return torch.where(u > -math.log(2.0),
                       torch.log(-torch.expm1(u)),
                       torch.log1p(-torch.exp(u)))


# --------------------------------------------------------------------------- #
# Cox PH — Breslow partial log-likelihood
# --------------------------------------------------------------------------- #
def cox_ph_loss(risk: Tensor, time: Tensor, event: Tensor, reduction: str = "events",
                ties: str = "breslow") -> Tensor:
    """Negative Cox partial log-likelihood.
    reduction: 'events' (mean per event, DeepSurv convention) | 'mean' (per sample) | 'sum'.
    ties: 'breslow' (fast, default — unchanged behaviour) | 'efron' (accurate under many tied
    event times, e.g. survival measured in whole months; matches lifelines' default).
    Efron reduces exactly to Breslow when there are no ties. The Cox partial likelihood is
    invariant to a global shift of `risk`, so we shift by max for numerical stability.
    """
    risk = risk.reshape(-1); time = time.reshape(-1); event = event.reshape(-1).float()
    if ties == "breslow":
        order = torch.argsort(time, descending=True)            # index 0 = largest time
        risk_ord, event_ord = risk[order], event[order]
        log_cum_hazard = torch.logcumsumexp(risk_ord, dim=0)    # log sum over risk set R(t_i)
        pll = (risk_ord - log_cum_hazard) * event_ord
        neg_ll, n_events = -pll.sum(), event_ord.sum()
    elif ties == "efron":
        r = risk - risk.max()                                   # shift-invariant, stable (r<=0)
        exp_r = torch.exp(r)
        ll = risk.new_zeros(())
        ev_times = torch.unique(time[event > 0])
        for t in ev_times:
            deaths = (time == t) & (event > 0)
            at_risk = time >= t
            m = int(deaths.sum())
            S_risk, S_death = exp_r[at_risk].sum(), exp_r[deaths].sum()
            l = torch.arange(m, device=risk.device, dtype=risk.dtype)
            denom = (S_risk - (l / m) * S_death).clamp_min(1e-12)
            ll = ll + r[deaths].sum() - torch.log(denom).sum()  # Efron partial log-lik
        neg_ll, n_events = -ll, event.sum()
    else:
        raise ValueError("ties must be 'breslow' or 'efron'")
    if reduction == "sum":
        return neg_ll
    if reduction == "mean":
        return neg_ll / risk.shape[0]
    return neg_ll / torch.clamp(n_events, min=1.0)


# --------------------------------------------------------------------------- #
# Log-normal AFT
# --------------------------------------------------------------------------- #
def lognormal_aft_loss(params: Tensor, time: Tensor, event: Tensor, reduction: str = "mean") -> Tensor:
    """Right-censored log-normal AFT NLL. params[:,0]=mu, params[:,1]=log_sigma."""
    mu = params[:, 0]
    sigma = torch.exp(params[:, 1].clamp(-5.0, 3.0))
    y = torch.log(time.reshape(-1).clamp_min(1e-8))
    event = event.reshape(-1).float()
    z = (y - mu) / sigma
    log_pdf = -0.5 * (_LOG_2PI + z * z) - torch.log(sigma) - y   # log f_T(t)
    log_surv = log_ndtr(-z)                                      # STABLE log S(t)=log Phi(-z)
    nll = -(event * log_pdf + (1.0 - event) * log_surv)
    return nll.mean() if reduction == "mean" else nll.sum()


# --------------------------------------------------------------------------- #
# Inverse-Gaussian first-hitting time
# --------------------------------------------------------------------------- #
def first_hitting_time_loss(params: Tensor, time: Tensor, event: Tensor, reduction: str = "mean") -> Tensor:
    """Right-censored IG first-passage NLL. params[:,0]=log_mu, params[:,1]=log_lambda.
    Survival computed in log-space (log_ndtr + log1mexp): NO exp(2*lam/mu) overflow,
    NO nan_to_num masking — finite by construction."""
    mu = torch.exp(params[:, 0].clamp(-5.0, 6.0))
    lam = torch.exp(params[:, 1].clamp(-5.0, 6.0))
    t = time.reshape(-1).clamp_min(1e-8)
    event = event.reshape(-1).float()
    log_pdf = 0.5 * (torch.log(lam) - _LOG_2PI - 3.0 * torch.log(t)) - lam * (t - mu) ** 2 / (2.0 * mu * mu * t)
    sqrt_lt = torch.sqrt(lam / t)
    a = sqrt_lt * (t / mu - 1.0)
    b = -sqrt_lt * (t / mu + 1.0)
    log_phi_na = log_ndtr(-a)
    u = (2.0 * lam / mu + log_ndtr(b) - log_phi_na).clamp_max(-1e-12)  # S>0 => u<0
    log_surv = log_phi_na + _log1mexp(u)
    nll = -(event * log_pdf + (1.0 - event) * log_surv)
    return nll.mean() if reduction == "mean" else nll.sum()


# --------------------------------------------------------------------------- #
# Risk + real predicted survival (kills the sigmoid-of-risk hack downstream)
# --------------------------------------------------------------------------- #
def risk_from_output(model_type: str, output: Tensor) -> Tensor:
    base = model_type.replace("opsd_", "")
    if base == "cox":
        return output.reshape(-1)
    return -output[:, 0].reshape(-1)   # AFT/FHT: larger location => longer time => lower risk


def survival_at(model_type: str, output: Tensor, horizon: float) -> Tensor:
    """Closed-form predicted S(horizon) in [0,1] for AFT/FHT (a real probability).
    Cox needs a Breslow baseline fit at eval time — handle there, not here."""
    base = model_type.replace("opsd_", "")
    if base == "aft":
        mu = output[:, 0]; sigma = torch.exp(output[:, 1].clamp(-5.0, 3.0))
        z = (math.log(max(float(horizon), 1e-8)) - mu) / sigma
        return torch.exp(log_ndtr(-z))
    if base == "fht":
        mu = torch.exp(output[:, 0].clamp(-5.0, 6.0)); lam = torch.exp(output[:, 1].clamp(-5.0, 6.0))
        t = torch.as_tensor(max(float(horizon), 1e-8), dtype=output.dtype, device=output.device)
        sqrt_lt = torch.sqrt(lam / t); a = sqrt_lt * (t / mu - 1.0); b = -sqrt_lt * (t / mu + 1.0)
        log_phi_na = log_ndtr(-a)
        u = (2.0 * lam / mu + log_ndtr(b) - log_phi_na).clamp_max(-1e-12)
        return torch.exp(log_phi_na + _log1mexp(u))
    raise ValueError("survival_at: Cox requires a fitted Breslow baseline (compute at eval).")


def event_prob_at(model_type: str, output: Tensor, horizon: float) -> Tensor:
    return 1.0 - survival_at(model_type, output, horizon)


# --------------------------------------------------------------------------- #
# Distillation (temperature removed; survival-curve variant added)
# --------------------------------------------------------------------------- #
def soft_distillation_loss(student_risk: Tensor, teacher_risk: Tensor) -> Tensor:
    """MSE between student and EMA-teacher risk (RANKING transfer only).
    Temperature removed: dividing by T then z-standardizing cancelled it exactly.
    Prefer survival_curve_distillation() for CALIBRATED transfer."""
    return torch.mean((student_risk - teacher_risk.detach()) ** 2)


def survival_curve_distillation(student_out: Tensor, teacher_out: Tensor,
                                model_type: str, grid) -> Tensor:
    """Distill the teacher's predicted survival CURVE into the student (transfers
    calibrated risk, not just ranking). grid: iterable of horizons. AFT/FHT only."""
    losses = []
    for h in grid:
        ss = survival_at(model_type, student_out, float(h))
        st = survival_at(model_type, teacher_out, float(h)).detach()
        losses.append(torch.mean((ss - st) ** 2))
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# Differentiable Cox survival-curve primitives (used by the HSS trainer's
# cross-head curve distillation: subtype head -> agnostic teacher).
# --------------------------------------------------------------------------- #
def cox_survival_curve(eta: Tensor, baseline_cum_hazard: Tensor) -> Tensor:
    """S(t_grid | eta) = exp(-H0(t_grid) * exp(eta)) as an [n, len(grid)] tensor.
    `baseline_cum_hazard` is the (Breslow) baseline cumulative hazard on the grid."""
    return torch.exp(-torch.outer(torch.exp(eta.reshape(-1)),
                                  baseline_cum_hazard.reshape(-1)))


def survival_curve_distill(s_student: Tensor, s_teacher: Tensor) -> Tensor:
    """MSE between two predicted survival curves (teacher detached)."""
    return torch.mean((s_student - s_teacher.detach()) ** 2)


def d_calibration_penalty(surv_at_obs: Tensor, event: Tensor, n_bins: int = 10,
                          bandwidth: float | None = None) -> Tensor:
    """Differentiable D-calibration / smoothed-ECE penalty (training-time analogue of the
    eval metrics_ipcw.d_calibration chi-square).

    Under a well-calibrated model the survival probabilities S_i(t_i) at the OBSERVED event
    times are Uniform(0,1). This soft-bins the event S values with a Gaussian kernel (so it is
    differentiable, unlike the hard-binned chi-square) and penalises deviation of the per-bin
    mass from the uniform target 1/n_bins. 0 ⇒ perfectly uniform. Apply to AFT/FHT heads (which
    give closed-form S(t)); weight it into the objective to pull calibration toward uniform.
    """
    s = surv_at_obs.reshape(-1).clamp(1e-6, 1 - 1e-6)
    ev = event.reshape(-1) > 0.5
    s_ev = s[ev]                                              # events must be ~Uniform(0,1)
    if s_ev.numel() < 2:
        return surv_at_obs.new_zeros(())
    centers = torch.linspace(1.0 / (2 * n_bins), 1 - 1.0 / (2 * n_bins), n_bins,
                             device=s.device, dtype=s.dtype)
    bw = bandwidth if bandwidth is not None else 1.0 / n_bins
    w = torch.exp(-((s_ev[:, None] - centers[None, :]) ** 2) / (2 * bw ** 2))
    w = w / w.sum(dim=1, keepdim=True)                        # each event spreads unit mass over bins
    frac = w.sum(dim=0) / s_ev.numel()                        # soft per-bin fraction
    return ((frac - 1.0 / n_bins) ** 2).sum()                # deviation from uniform