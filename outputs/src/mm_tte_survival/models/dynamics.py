"""Latent-time dynamics for ResistanceMap (Part II of the dynamic-upgrade roadmap).

The dashboard's L2 "when" gap: today the resistance state is static — the survival head
reads a single fused latent. Here the latent is allowed to *evolve* along a disease
latent-time τ∈[0,1], and (Stage 4) the event becomes its first passage across a barrier.

Two implementations behind one ABC:
  * :class:`StaticDynamics` — the backward-compatibility anchor. v≡0, flow is the identity,
    so ``ResistanceModel(..., dynamics=StaticDynamics(d))`` reproduces today's model exactly.
  * :class:`FlowMatchingDynamics` — a SiLU velocity field with a Fourier time embedding,
    integrated by explicit Euler (no adjoint; CFM is simulation-free — see training/flow_losses).

Nothing here is trained yet; Stage 2 adds the CFM loss, Stage 3 the public stage marginals,
Stage 4 the flow→FHT coupling. The field is a population/stage-level trajectory (honest scope).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


def _time_embed(tau: Tensor, dim: int = 16) -> Tensor:
    """Fourier features of latent-time tau∈[0,1] -> (N, dim)."""
    freqs = torch.exp(torch.linspace(0, math.log(1e3), dim // 2, device=tau.device))
    a = tau.reshape(-1, 1) * freqs
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class LatentDynamics(nn.Module, ABC):
    """v(τ, z, c): the velocity of the disease latent at latent-time τ under conditioning c."""

    latent_dim: int

    @abstractmethod
    def velocity(self, z: Tensor, c: Tensor, tau: Tensor) -> Tensor:
        """Instantaneous velocity field, shape (N, latent_dim)."""

    @abstractmethod
    def flow(self, z0: Tensor, c: Tensor, tau_max: Tensor) -> tuple[Tensor, Tensor]:
        """Integrate z from τ=0 to τ=tau_max. Returns (z(tau_max), mean velocity over the path).
        Cheap fixed-step, no adjoint."""


class StaticDynamics(LatentDynamics):
    """Identity dynamics: recovers the current static ResistanceModel exactly (v≡0)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def velocity(self, z: Tensor, c: Tensor, tau: Tensor) -> Tensor:
        return torch.zeros_like(z)

    def flow(self, z0: Tensor, c: Tensor, tau_max: Tensor) -> tuple[Tensor, Tensor]:
        return z0, torch.zeros_like(z0)


class FlowMatchingDynamics(LatentDynamics):
    """Learned velocity field over disease latent-time, trained by OT-CFM (Stage 2+)."""

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int = 256,
                 n_steps: int = 8, time_dim: int = 16):
        super().__init__()
        self.latent_dim, self.cond_dim = latent_dim, cond_dim
        self.n_steps, self.time_dim = n_steps, time_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim))

    def velocity(self, z: Tensor, c: Tensor, tau: Tensor) -> Tensor:
        te = _time_embed(tau.reshape(-1), self.time_dim).to(z.dtype)
        return self.net(torch.cat([z, c, te], dim=-1))

    def flow(self, z0: Tensor, c: Tensor, tau_max: Tensor) -> tuple[Tensor, Tensor]:
        n = self.n_steps
        tau_max = tau_max.reshape(-1, 1)
        z, v_sum = z0, torch.zeros_like(z0)
        for k in range(n):
            tau = (k / n) * tau_max
            v = self.velocity(z, c, tau)
            z = z + (tau_max / n) * v          # explicit Euler
            v_sum = v_sum + v
        return z, v_sum / n                     # mean velocity over the path


__all__ = ["LatentDynamics", "StaticDynamics", "FlowMatchingDynamics"]
