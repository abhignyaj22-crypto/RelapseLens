"""Latent-crossing head: Brownian first-passage -> inverse-Gaussian FHT params.

The scalar disease-burden readout s_t = w_cᵀ z_t is modelled as Brownian motion with drift
ν>0 and volatility σ>0 starting a gap = a_c - s_0 > 0 below the barrier a_c. The first
passage time of s_t to a_c is Inverse-Gaussian:

    T ~ IG(mean μ = gap/ν,  shape λ = gap² / σ²).

We emit (log μ, log λ) so the existing, scipy-verified `first_hitting_time_loss` /
`survival_at("fht", ...)` in training/losses.py consume it unchanged.

Identifiability: the readout weight w_c is constrained NON-NEGATIVE (softplus) so an increase
in the latent state always increases burden ("crossing up = worsening"). Without this, the
barrier a_c and w_c are jointly non-identified up to a sign flip.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import log_ndtr

from ..training.losses import _log1mexp

_EPS = 1e-6
_LOG_2PI = math.log(2.0 * math.pi)


def fht_params_from_crossing(gap: torch.Tensor, drift: torch.Tensor,
                             sigma: torch.Tensor) -> torch.Tensor:
    """Map Brownian first-passage geometry to inverse-Gaussian (log_mu, log_lambda).

    PROPER-CASE (drift > 0) ONLY. mu = gap/drift is singular/negative for drift <= 0, so this
    (and `CrossingHead` / `training.losses.first_hitting_time_loss`, which clamp mu > 0) must
    NOT be used to score responder-bearing data — those drift-<=0 patients need the defective
    path (`crossing_nll` / `crossing_log_survival` / `never_cross_prob`). This proper form is
    retained only for the scipy closed-form check and future strictly-progressing use.

    gap = a_c - s_0 (> 0), drift ν (> 0), volatility σ (> 0). Returns an (N, 2) tensor
    [log μ, log λ] with μ = gap/ν and λ = gap²/σ², matching the parameterisation of
    `training.losses.first_hitting_time_loss`.
    """
    gap = gap.reshape(-1).clamp_min(_EPS)
    drift = drift.reshape(-1).clamp_min(_EPS)
    var = (sigma.reshape(-1) ** 2).clamp_min(_EPS)
    log_mu = torch.log(gap) - torch.log(drift)
    log_lambda = 2.0 * torch.log(gap) - torch.log(var)
    return torch.stack([log_mu, log_lambda], dim=1)


# --------------------------------------------------------------------------- #
# Defective first-passage: Brownian motion with drift of ANY sign to barrier a>0.
# For drift <= 0 (a responder whose burden is falling) the first-passage law is a
# DEFECTIVE inverse-Gaussian with mass at +infinity: P(never cross) = 1 - exp(2 v a / s^2).
# Parameterised directly by (gap, drift, sigma) so there is NO division by drift and
# no mu = gap/drift singularity at drift -> 0. This is the honest handling the existing
# first_hitting_time_loss (which clamps mu>0) cannot express.
# --------------------------------------------------------------------------- #
def _std_args(gap, drift, sigma):
    a = torch.as_tensor(gap).reshape(-1).clamp_min(_EPS)
    v = torch.as_tensor(drift).reshape(-1)
    s2 = (torch.as_tensor(sigma).reshape(-1) ** 2).clamp_min(_EPS)
    return a, v, s2


def crossing_log_survival(gap, drift, sigma, t) -> torch.Tensor:
    """log P(not yet crossed by t) for BM(drift, sigma) to barrier gap>0, any drift sign.

    S(t) = Phi((a - v t)/(s sqrt t)) - exp(2 v a / s^2) Phi((-a - v t)/(s sqrt t)),
    evaluated stably in log-space (log_ndtr + log1mexp)."""
    a, v, s2 = _std_args(gap, drift, sigma)
    s = torch.sqrt(s2)
    t = torch.as_tensor(t, dtype=a.dtype, device=a.device).reshape(-1).clamp_min(_EPS)
    sqrt_t = torch.sqrt(t)
    z1 = (a - v * t) / (s * sqrt_t)
    z2 = (-a - v * t) / (s * sqrt_t)
    log_phi1 = log_ndtr(z1)
    u = (2.0 * v * a / s2 + log_ndtr(z2) - log_phi1).clamp_max(-1e-12)  # S>0 => u<0
    return log_phi1 + _log1mexp(u)


def crossing_log_pdf(gap, drift, sigma, t) -> torch.Tensor:
    """log first-passage density (the sub-density; integrates to P(ever cross) <= 1)."""
    a, v, s2 = _std_args(gap, drift, sigma)
    t = torch.as_tensor(t, dtype=a.dtype, device=a.device).reshape(-1).clamp_min(_EPS)
    return torch.log(a) - 0.5 * (_LOG_2PI + torch.log(s2) + 3.0 * torch.log(t)) \
        - (a - v * t) ** 2 / (2.0 * s2 * t)


def never_cross_prob(gap, drift, sigma) -> torch.Tensor:
    """P(never cross) = 1 - exp(2 v a / s^2) for drift<0, else 0 (drift>=0 crosses a.s.)."""
    a, v, s2 = _std_args(gap, drift, sigma)
    p_ever = torch.where(v < 0.0, torch.exp(2.0 * v * a / s2), torch.ones_like(v))
    return (1.0 - p_ever).clamp(0.0, 1.0)


def crossing_risk_score(gap, drift, sigma, horizon: float) -> torch.Tensor:
    """Risk = P(cross by `horizon`) = 1 - S(horizon). Higher => earlier event. Responders
    (drift<=0) land near their non-crossing floor, so they rank low-risk by construction."""
    return 1.0 - torch.exp(crossing_log_survival(gap, drift, sigma, horizon))


def crossing_nll(gap, drift, sigma, t, event) -> torch.Tensor:
    """Right-censored defective-FHT NLL. event=1 -> -log f(t); event=0 -> -log S(t).
    A censored responder contributes -log S which tends to -log(P(never cross)) as t grows,
    i.e. the cure fraction is learned, not clamped away."""
    a, v, s2 = _std_args(gap, drift, sigma)
    s = torch.sqrt(s2)
    e = torch.as_tensor(event, dtype=a.dtype, device=a.device).reshape(-1)
    log_f = crossing_log_pdf(a, v, s, t)
    log_s = crossing_log_survival(a, v, s, t)
    return -(e * log_f + (1.0 - e) * log_s)


class MonotoneReadout(nn.Module):
    """s = wᵀ z with w ≥ 0 (softplus-reparameterised) so ∂s/∂z ≥ 0 elementwise."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.raw_w = nn.Parameter(torch.zeros(latent_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    @property
    def weight(self) -> torch.Tensor:
        return F.softplus(self.raw_w)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.weight + self.bias  # (N,)


class CrossingHead(nn.Module):
    """Turn a per-patient latent-path summary into inverse-Gaussian FHT params.

    PROPER-CASE (drift > 0) ONLY — emits (log μ, log λ) via `fht_params_from_crossing` for the
    clamped `first_hitting_time_loss`. It cannot represent the defective law of a responder
    (drift <= 0); the Stage-1 teacher scores through the defective functional path
    (`crossing_nll` / `crossing_risk_score`) instead. Kept for strictly-progressing use.

    Inputs are the drift ν and volatility σ of the readout s_t (supplied by the state-space
    in Stage 1) plus the current burden s_0; the barrier a_c is a positive learnable scalar
    (optionally per cytogenetic stratum). Emits (log μ, log λ) for the FHT loss.
    """

    def __init__(self, n_strata: int = 1, init_barrier: float = 1.0):
        super().__init__()
        # softplus(raw_barrier) keeps a_c > 0; one per stratum (shared model => n_strata=1).
        inv = torch.log(torch.expm1(torch.tensor(float(init_barrier))))
        self.raw_barrier = nn.Parameter(inv.repeat(n_strata))

    def barrier(self, stratum: torch.Tensor | None = None) -> torch.Tensor:
        a = F.softplus(self.raw_barrier)
        if stratum is None:
            return a[0].expand(1)
        return a[stratum.reshape(-1)]

    def forward(self, s0: torch.Tensor, drift: torch.Tensor, sigma: torch.Tensor,
                stratum: torch.Tensor | None = None) -> torch.Tensor:
        a = self.barrier(stratum)
        if a.numel() == 1:
            a = a.expand_as(s0.reshape(-1))
        gap = a - s0.reshape(-1)
        return fht_params_from_crossing(gap, drift, sigma)
