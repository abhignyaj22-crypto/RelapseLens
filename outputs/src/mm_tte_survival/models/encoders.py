"""Polymorphic per-modality encoders — all return (N, d_latent) for fusion.

Distinct from `mlp_encoder.py` (the shared MLP building block this file composes): here each subclass of
`ModalityEncoder` adapts one modality (transcriptome, program-activity, VAE, PPI graph, scRNA state).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .mlp_encoder import MLPEncoder, build_encoder


class ModalityEncoder(nn.Module, ABC):
    """Encoder interface: batch in → latent out."""

    latent_dim: int

    @abstractmethod
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Return (N, d_latent)."""


class TranscriptomeEncoder(ModalityEncoder):
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = build_encoder(input_dim, {"type": "mlp", "latent_dim": latent_dim,
                                              "hidden_dims": [hidden], "dropout": dropout})

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch)


class ProgramActivityEncoder(ModalityEncoder):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch)


class VAEEncoder(ModalityEncoder):
    """Cell-line proteome→epigenome VAE encoder (L1 placeholder — Stage D)."""

    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.mu(batch)


class PPIGraphEncoder(ModalityEncoder):
    """PPI-GNN encoder placeholder (L3 — Stage D)."""

    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(input_dim, latent_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch)


class ScRNAStateEncoder(ModalityEncoder):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = MLPEncoder(input_dim, latent_dim, hidden_dims=(64,))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch)
