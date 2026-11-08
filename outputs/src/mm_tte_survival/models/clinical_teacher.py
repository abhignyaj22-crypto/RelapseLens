"""Stage 2 / PR2 scaffold — clinically anchored longitudinal FHT teacher (data-gated).

This is the "when" core of the longitudinal redesign: a time-aware encoder over a patient's serial
clinical history produces a latent progression state z(t_k) at the landmark index; a calendar-time
`ClinicalAnchoredDynamics` field rolls it forward; and a *defective* first-hitting-time readout
(`longitudinal.crossing_head`) turns it into horizon crossing risks {p_90,p_180,p_365} plus a learned
non-crossing mass P(T*=∞)>0 for responders.  The teacher emits exactly the target dict PR4 distillation
consumes: q^T = {z_tk, s0, drift, sigma, p_h…, P_inf}.

SCIENTIFIC GATE (non-negotiable, mirrors Stage 1): `ClinicalTeacher.fit` / `assert_trainable_on` call
`data.serial_clinical.require_real_serial_substrate` FIRST and raise `SubstrateGateError` on synthetic,
demo, or baseline-only (1-visit) data.  The forward pass runs on any tensor so shapes/monotonicity can be
unit-tested, but **no training may occur without a real, dense serial substrate** — which is not on the
open GDC tier today (see docs/paper/progression_potential_design.md).  Nothing here licenses a claim.

Distinct from `FlowMatchingDynamics` (population / latent-time, OT-CFM): this dynamics is CALENDAR-time
and patient-anchored — kept separate so a latent τ is never silently reported as calendar t.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .dynamics import LatentDynamics, _time_embed
from ..longitudinal.crossing_head import (
    MonotoneReadout, crossing_nll, crossing_risk_score, never_cross_prob)

DEFAULT_HORIZONS_DAYS: tuple[float, ...] = (90.0, 180.0, 365.0)


# --------------------------------------------------------------------------- dynamics
class ClinicalAnchoredDynamics(LatentDynamics):
    """Calendar-time latent drift anchored at the last visit: dz/dt = f_θ(z, c, t).

    Semantics is calendar_time (days), NOT latent pseudotime — asserted via the `semantics` tag so the
    claim-gate can forbid a calendar horizon on a pseudotime axis.  Fixed-step Euler, no adjoint.
    """

    semantics = "calendar_time"

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int = 128,
                 n_steps: int = 8, time_dim: int = 16):
        super().__init__()
        self.latent_dim, self.cond_dim = latent_dim, cond_dim
        self.n_steps, self.time_dim = n_steps, time_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim))

    def velocity(self, z: Tensor, c: Tensor, tau: Tensor) -> Tensor:
        te = _time_embed(tau.reshape(-1), self.time_dim).to(z.dtype)
        return self.net(torch.cat([z, c, te], dim=-1))

    def flow(self, z0: Tensor, c: Tensor, tau_max: Tensor) -> tuple[Tensor, Tensor]:
        n = self.n_steps
        tmax = tau_max.reshape(-1, 1)
        z, v_sum = z0, torch.zeros_like(z0)
        for i in range(n):
            tau = tmax.squeeze(-1) * (i / n)
            v = self.velocity(z, c, tau)
            z = z + v * (tmax / n)
            v_sum = v_sum + v
        return z, v_sum / n


# --------------------------------------------------------------------------- encoder
class ClinicalVisitEncoder(nn.Module):
    """Time-aware GRU over irregularly sampled visits → latent state at the last observed visit.

    Input per visit is [labs, treatment-context, Δt-since-previous]; the Δt channel makes the recurrence
    gap-aware (a compact GRU-D-style stand-in — deliberately small for identifiability at MM cohort N,
    per the roadmap: "not a giant transformer").  `mask` (N,K) marks valid visits; padded steps are
    zeroed and the last *valid* hidden state is selected per patient.
    """

    def __init__(self, n_labs: int, n_treat: int, latent_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.in_dim = n_labs + n_treat + 1  # +1 for Δt
        self.gru = nn.GRU(self.in_dim, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, labs: Tensor, treat: Tensor, dt: Tensor, mask: Tensor) -> Tensor:
        x = torch.cat([labs, treat, dt.unsqueeze(-1)], dim=-1) * mask.unsqueeze(-1)
        h, _ = self.gru(x)                                  # (N,K,hidden)
        last = mask.long().sum(dim=1).clamp(min=1) - 1      # index of last valid visit
        h_last = h[torch.arange(h.size(0)), last]
        return self.to_latent(h_last)


# --------------------------------------------------------------------------- teacher
@dataclass
class TeacherOutput:
    z_tk: Tensor       # latent state at landmark index (N, latent)
    s0: Tensor         # current monotone burden readout (N,)
    drift: Tensor      # ν of the readout (N,) — may be <=0 (responder)
    sigma: Tensor      # volatility σ>0 (N,)
    horizons: tuple[float, ...]
    risk: dict[float, Tensor]   # horizon(days) -> P(cross by h) in [0,1]
    p_never: Tensor    # P(T*=∞) non-crossing mass (N,)


class ClinicalTeacher(nn.Module):
    """Encoder → z(t_k) → (drift, σ) head → defective-FHT horizon risks + non-crossing mass.

    The (drift, σ) head plays the state-space role the crossing head expects; `ClinicalAnchoredDynamics`
    is attached for the flow-coupled variant (PR2 headline experiment) and to keep the calendar-time
    semantics explicit, but the scaffold's forward uses the direct head so shapes are testable without a
    trained flow.  Drift is unconstrained in sign (responders get ν≤0 → mass piles on P(T*=∞)).
    """

    def __init__(self, n_labs: int, n_treat: int, cond_dim: int,
                 latent_dim: int = 32, hidden: int = 64,
                 horizons_days: tuple[float, ...] = DEFAULT_HORIZONS_DAYS,
                 init_barrier: float = 1.0):
        super().__init__()
        self.encoder = ClinicalVisitEncoder(n_labs, n_treat, latent_dim, hidden)
        self.dynamics = ClinicalAnchoredDynamics(latent_dim, cond_dim)
        self.readout = MonotoneReadout(latent_dim)
        self.drift_sigma = nn.Linear(latent_dim + cond_dim, 2)
        inv = torch.log(torch.expm1(torch.tensor(float(init_barrier))))
        self.raw_barrier = nn.Parameter(inv)       # a_c > 0 via softplus
        self.horizons = tuple(float(h) for h in horizons_days)

    @property
    def barrier(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_barrier)

    def forward(self, labs: Tensor, treat: Tensor, dt: Tensor, mask: Tensor, c: Tensor) -> TeacherOutput:
        z = self.encoder(labs, treat, dt, mask)
        s0 = self.readout(z)
        ds = self.drift_sigma(torch.cat([z, c], dim=-1))
        drift = ds[:, 0]
        sigma = torch.nn.functional.softplus(ds[:, 1]) + 1e-3
        gap = (self.barrier - s0).reshape(-1)
        risk = {h: crossing_risk_score(gap, drift, sigma, h) for h in self.horizons}
        return TeacherOutput(z_tk=z, s0=s0, drift=drift, sigma=sigma, horizons=self.horizons,
                             risk=risk, p_never=never_cross_prob(gap, drift, sigma))

    def crossing_loss(self, out: TeacherOutput, event_t_days: Tensor, event: Tensor) -> Tensor:
        """Right-censored defective-FHT NLL at the observed event/censor time."""
        gap = (self.barrier - out.s0).reshape(-1)
        return crossing_nll(gap, out.drift, out.sigma, event_t_days, event).mean()

    # -------------------------------------------------- the training gate (mirrors Stage 1)
    @staticmethod
    def assert_trainable_on(visit_table, **gate_kwargs) -> None:
        """Raise SubstrateGateError unless `visit_table` is a REAL, dense serial substrate.

        Called at the top of any fit loop.  Synthetic/demo (is_real=False) and baseline-only (1-visit)
        tables fail here — the teacher cannot be trained, and therefore cannot make a claim, on data
        that does not exist on the open tier.
        """
        from ..data.serial_clinical import require_real_serial_substrate
        require_real_serial_substrate(visit_table, **gate_kwargs)

    def fit(self, visit_table, *args, **kwargs):
        """Placeholder trainer — GATE FIRST.  Real optimisation lands in PR2 proper once a real serial
        substrate is on disk; today this deliberately refuses rather than train on absent/fake data."""
        self.assert_trainable_on(visit_table)
        raise NotImplementedError(
            "ClinicalTeacher.fit: substrate gate passed but the PR2 training loop is intentionally "
            "unbuilt until real serial CoMMpass data is present. Scaffold only.")
