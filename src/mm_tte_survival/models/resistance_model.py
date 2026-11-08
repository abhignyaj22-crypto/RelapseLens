"""ResistanceModel — the composed encoders → fusion → head model.

This is the reconciliation keystone: ResistanceMap's representation stack (per-modality
:class:`ModalityEncoder`s) feeds a :class:`FusionModule`, whose fused latent feeds an
MMTTE :class:`SurvivalHead`. Swapping Cox↔AFT↔FHT or concat↔attention↔residual fusion
is a construction choice, not a code fork — that is the payoff of the five ABCs.

Two interpretability tiers coexist (the user-chosen "1 & 2" refactor):
  * **Canonical / interpretable headline** — :class:`ResidualRiskModel` (re-exported
    here unchanged): a lifelines additive Cox with the EXACT ``total = clinical +
    molecular_residual`` identity. It remains the manuscript headline and its 104-test
    behaviour is untouched.
  * **Neural composition** — :class:`ResistanceModel` with :class:`ResidualFusion` +
    :class:`CoxHead` is the latent-space analogue of that identity and the experimental
    path that must *beat* the headline on paired ΔC (Stage E) to earn promotion.

``ResistanceModel.interpretable_residual(...)`` returns the canonical
``ResidualRiskModel`` so both are reachable through one entry point.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .dynamics import LatentDynamics, StaticDynamics
from .encoders import ModalityEncoder
from .fusion import FusionModule, ResidualFusion
from .heads import SurvivalHead
from .residual_risk import ResidualRiskModel  # canonical interpretable headline (unchanged)


class ResistanceModel(nn.Module):
    """encoders(dict) -> fusion -> survival head. Domain-honest multimodal composition.

    Optional latent-time ``dynamics`` (Part II): when a ``dynamics`` and a ``horizon_tau``
    are supplied the fused latent is flowed along disease latent-time before the head sees
    it. With ``dynamics=None`` (default) or ``StaticDynamics`` or no ``horizon_tau``, the
    forward pass is the current static model **bit-for-bit** — the compatibility anchor.
    """

    def __init__(self, encoders: dict[str, ModalityEncoder], fusion: FusionModule,
                 head: SurvivalHead, dynamics: LatentDynamics | None = None,
                 readout: Tensor | None = None, barrier: Tensor | None = None):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion
        self.head = head
        self.dynamics = dynamics          # None => pure static (today's behaviour)
        self.readout = readout            # w_c for the dynamic FHT coupling (Stage 4)
        self.barrier = barrier            # a_c barrier (Stage 4)

    def encode(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Per-modality latents for every modality present in the batch."""
        return {name: enc(batch[name]) for name, enc in self.encoders.items() if name in batch}

    def _is_static(self, horizon_tau) -> bool:
        return self.dynamics is None or isinstance(self.dynamics, StaticDynamics) or horizon_tau is None

    def forward(self, batch: dict[str, Tensor], c: Tensor | None = None,
                horizon_tau: Tensor | None = None) -> dict[str, Tensor]:
        latents = self.encode(batch)
        z = self.fusion(latents)
        if self._is_static(horizon_tau):
            params = self.head(z)
            return {"z": z, "params": params, "risk": self.head.risk(params),
                    "model_type": self.head.model_type}
        # dynamic branch: flow the latent along disease latent-time
        z_tau, v_mean = self.dynamics.flow(z, c, horizon_tau)
        return {"z0": z, "z_tau": z_tau, "v_mean": v_mean}

    def nll(self, batch: dict[str, Tensor], time: Tensor, event: Tensor) -> Tensor:
        return self.head.nll(self.forward(batch)["params"], time, event)

    @staticmethod
    def interpretable_residual(clinical_cols, molecular_cols, **kw) -> ResidualRiskModel:
        """Return the canonical interpretable headline (lifelines additive Cox)."""
        return ResidualRiskModel(clinical_cols=clinical_cols, molecular_cols=molecular_cols, **kw)


__all__ = ["ResistanceModel", "ResidualRiskModel", "ResidualFusion",
           "LatentDynamics", "StaticDynamics"]
