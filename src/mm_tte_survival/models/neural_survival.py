"""Abstract multi-head survival model: shared encoder + Cox/AFT/FHT heads.

Kept as a parallel experimental model. The production/default model is
ResidualRiskModel; OPSD distillation on this net is optional and claim-gated
(retained only if it improves calibration / repeated-split stability / paired ΔC).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mlp_encoder import MLPEncoder, build_encoder
from .survival_heads import build_survival_heads


class MultiHeadSurvivalModel(nn.Module):
    def __init__(self, encoder: MLPEncoder, survival_heads: nn.ModuleDict):
        super().__init__()
        self.encoder = encoder
        self.survival_heads = survival_heads

    def forward(self, batch: torch.Tensor) -> dict:
        z = self.encoder(batch)
        out = {"z": z}
        for name, head in self.survival_heads.items():
            out[name] = head(z)
        return out


def build_multihead_model(input_dim: int, encoder_cfg: dict | None = None,
                          heads=("cox", "aft", "fht")) -> MultiHeadSurvivalModel:
    enc = build_encoder(input_dim, encoder_cfg or {})
    return MultiHeadSurvivalModel(enc, build_survival_heads(enc.latent_dim, heads))
