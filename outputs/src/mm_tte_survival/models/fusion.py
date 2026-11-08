"""FusionModule — Strategy pattern for combining per-modality latents.

One of the five OOP base classes. A ``FusionModule`` maps a dict of per-modality
latents ``{name: (N, d_m)}`` to a single fused ``(N, d_out)`` representation. The
fusion never knows *which* modalities exist: missing or cell-line-transferred blocks
are handled by modality dropout (train-time) and mean-imputation (eval-time), so the
same composed model runs whether or not a given block is present.

Three strategies:
  * :class:`LateConcatFusion` — concat present latents (missing → learned/zero pad) then project.
  * :class:`CrossAttentionFusion` — multi-head attention over stacked modality tokens
    with modality dropout; the "when"/mechanism-aware experimental path.
  * :class:`ResidualFusion` — the *interpretable* strategy: a designated clinical block
    passes through, and every other block enters only as an additive residual over it,
    mirroring the exact ``clinical + molecular_residual`` identity of
    :class:`~mm_tte_survival.models.residual_risk.ResidualRiskModel` in latent space.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class FusionModule(nn.Module, ABC):
    """Abstract fusion: dict of modality latents -> (N, out_dim)."""

    out_dim: int

    @abstractmethod
    def forward(self, latents: dict[str, Tensor]) -> Tensor:
        ...

    @staticmethod
    def _stack(latents: dict[str, Tensor]) -> tuple[list[str], Tensor]:
        """Deterministic (sorted) stack of modality latents -> (N, M, d). Requires
        equal per-modality latent dim (project upstream if they differ)."""
        names = sorted(latents)
        return names, torch.stack([latents[n] for n in names], dim=1)


class LateConcatFusion(FusionModule):
    """Concatenate the (sorted) modality latents and project to out_dim."""

    def __init__(self, modality_dims: dict[str, int], out_dim: int = 64):
        super().__init__()
        self.modality_dims = dict(sorted(modality_dims.items()))
        self.out_dim = out_dim
        self.proj = nn.Linear(sum(self.modality_dims.values()), out_dim)

    def forward(self, latents: dict[str, Tensor]) -> Tensor:
        parts = []
        n = next(iter(latents.values())).shape[0]
        for name, d in self.modality_dims.items():
            if name in latents:
                parts.append(latents[name])
            else:  # missing block -> zero pad (present in schema, absent at runtime)
                parts.append(torch.zeros(n, d, device=self.proj.weight.device))
        return self.proj(torch.cat(parts, dim=1))


class CrossAttentionFusion(FusionModule):
    """Multi-head self-attention over modality tokens, with modality dropout.

    Each modality latent is projected to a common ``d_model`` token; attention lets
    the survival signal attend across modalities; the fused vector is the mean over
    surviving tokens. ``modality_dropout`` randomly drops whole modality tokens during
    training so the model is robust to missing/transferred blocks at inference."""

    def __init__(self, modality_dims: dict[str, int], d_model: int = 64,
                 n_heads: int = 4, modality_dropout: float = 0.2, out_dim: int | None = None):
        super().__init__()
        self.modality_dims = dict(sorted(modality_dims.items()))
        self.d_model = d_model
        self.out_dim = out_dim or d_model
        self.modality_dropout = modality_dropout
        self.proj = nn.ModuleDict({n: nn.Linear(d, d_model) for n, d in self.modality_dims.items()})
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Linear(d_model, self.out_dim)

    def forward(self, latents: dict[str, Tensor]) -> Tensor:
        names = [n for n in self.modality_dims if n in latents]
        tokens = torch.stack([self.proj[n](latents[n]) for n in names], dim=1)  # (N, M, d)
        n, m, _ = tokens.shape
        mask = None
        if self.training and self.modality_dropout > 0 and m > 1:
            keep = (torch.rand(n, m, device=tokens.device) >= self.modality_dropout)
            keep[~keep.any(dim=1), 0] = True  # never drop every modality for a sample
            mask = ~keep  # True == ignore (key_padding_mask convention)
        attended, _ = self.attn(tokens, tokens, tokens, key_padding_mask=mask)
        if mask is not None:
            w = (~mask).float().unsqueeze(-1)
            fused = (attended * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        else:
            fused = attended.mean(dim=1)
        return self.out(fused)


class ResidualFusion(FusionModule):
    """Interpretable additive fusion: clinical block passes through; every other
    block contributes only an additive residual over it (latent-space analogue of the
    exact clinical + molecular_residual identity). Keeps the headline model's
    attributability while living behind the uniform FusionModule interface."""

    def __init__(self, modality_dims: dict[str, int], clinical_key: str = "clinical",
                 out_dim: int = 64):
        super().__init__()
        self.modality_dims = dict(sorted(modality_dims.items()))
        self.clinical_key = clinical_key
        self.out_dim = out_dim
        if clinical_key not in self.modality_dims:
            raise ValueError(f"ResidualFusion needs a '{clinical_key}' modality")
        self.clinical_proj = nn.Linear(self.modality_dims[clinical_key], out_dim)
        self.residual_proj = nn.ModuleDict({
            n: nn.Linear(d, out_dim) for n, d in self.modality_dims.items() if n != clinical_key
        })

    def forward(self, latents: dict[str, Tensor]) -> Tensor:
        base = self.clinical_proj(latents[self.clinical_key])
        residual = torch.zeros_like(base)
        for name, proj in self.residual_proj.items():
            if name in latents:
                residual = residual + proj(latents[name])
        return base + residual  # additive: clinical baseline + molecular residual


__all__ = ["FusionModule", "LateConcatFusion", "CrossAttentionFusion", "ResidualFusion"]
