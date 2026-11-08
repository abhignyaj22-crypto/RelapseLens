"""Shared plain-MLP latent encoder building block (was `encoder.py`; renamed in A1 1.3 to end the
`encoder.py` vs `encoders.py` collision). Composed by `encoders.py` and `neural_survival.py` — this is the
generic block, `encoders.py` holds the per-modality subclasses."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64,
                 hidden_dims=(128, 64), dropout: float = 0.1):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], latent_dim)]
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_encoder(input_dim: int, cfg: dict | None = None, **kwargs) -> MLPEncoder:
    cfg = dict(cfg or {})
    cfg.update(kwargs)
    return MLPEncoder(
        input_dim=input_dim,
        latent_dim=int(cfg.get("latent_dim", 64)),
        hidden_dims=tuple(cfg.get("hidden_dims", (128, 64))),
        dropout=float(cfg.get("dropout", 0.1)),
    )
