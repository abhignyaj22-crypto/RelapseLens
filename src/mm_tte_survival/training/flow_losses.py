"""Conditional Flow Matching loss + exact minibatch-OT coupling + the flow→FHT bridge.

Part II Stage 2. The only genuinely new numerical surface in the whole dynamic upgrade is
here: the CFM regression loss, the OT coupling, and (for Stage 4) the coupling of the flow's
drift into the ALREADY scipy-verified defective first-passage head (crossing_nll). No new
first-passage math is introduced — dynamic_fht_nll simply feeds a flow-derived drift into it.

Two-part identification (resolves latent-time ↔ calendar-time):
  * conditional_flow_matching_loss learns the DIRECTION/shape of progression unsupervised
    from public stage marginals (ρ_HD→ρ_MGUS→ρ_SMM→ρ_MM);
  * dynamic_fht_nll identifies the calendar-time RATE from the survival labels (drift ν is
    in per-month units because `time` is in months).
The flow supplies the field; the survival head supplies the clock.
"""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment  # exact OT for equal minibatches (no new dep)
from torch import Tensor

from ..longitudinal.crossing_head import crossing_nll


def ot_couple(z_a: Tensor, z_b: Tensor) -> Tensor:
    """Reorder z_b to its exact squared-Euclidean OT assignment with z_a (equal sizes).

    Makes the straight-line CFM path the minimum-kinetic-energy geodesic (OT-CFM)."""
    if z_a.shape[0] != z_b.shape[0]:
        raise ValueError(f"ot_couple needs equal batch sizes, got {z_a.shape[0]} vs {z_b.shape[0]}")
    cost = torch.cdist(z_a, z_b) ** 2
    _, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return z_b[torch.as_tensor(col, device=z_b.device, dtype=torch.long)]


def conditional_flow_matching_loss(dyn, z_a: Tensor, z_b: Tensor, c: Tensor,
                                   sigma: float = 0.1, ot: bool = True) -> Tensor:
    """OT-CFM regression loss transporting stage marginal ρ_a → ρ_b.

        z_τ = (1-τ)·z_a + τ·z_b (+ σ·ε);  target u = z_b - z_a
        L    = E_{τ,pairs} || v_θ(τ, z_τ, c) - u ||²
    """
    if ot:
        z_b = ot_couple(z_a, z_b)
    tau = torch.rand(z_a.shape[0], 1, device=z_a.device)
    z_tau = (1 - tau) * z_a + tau * z_b
    if sigma > 0:
        z_tau = z_tau + sigma * torch.randn_like(z_tau)
    target = z_b - z_a
    return ((dyn.velocity(z_tau, c, tau) - target) ** 2).mean()


def dynamic_fht_nll(dyn, z0: Tensor, c: Tensor, readout_w: Tensor, barrier_a: Tensor,
                    horizon_tau: Tensor, time: Tensor, event: Tensor, sigma: Tensor) -> Tensor:
    """Couple the flow to the event: the resistance readout s_t = wᵀz_t drifts under the
    learned field; the event is its first passage across barrier a_c. Reuses the verified
    defective crossing NLL so responders (drift ≤ 0) get an honest cure fraction — no new math.
    """
    _, v_mean = dyn.flow(z0, c, horizon_tau)          # mean velocity over the path
    drift = (v_mean * readout_w).sum(-1)              # ν = wᵀ v_mean  (per-month rate)
    gap = (barrier_a - (z0 * readout_w).sum(-1)).clamp_min(1e-6)
    return crossing_nll(gap, drift, sigma, time, event).mean()


__all__ = ["ot_couple", "conditional_flow_matching_loss", "dynamic_fht_nll"]
