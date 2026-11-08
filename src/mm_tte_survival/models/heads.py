"""SurvivalHead — the MMTTE survival-head family under one polymorphic interface.

One of the five OOP base classes of the ResistanceMap × MMTTE reconciliation. A
``SurvivalHead`` maps a shared latent ``z`` to distribution parameters (``forward``)
and exposes a uniform likelihood / survival / risk contract so the fusion layer and
the composed :class:`~mm_tte_survival.models.resistance_model.ResistanceModel` never
need to know *which* head they carry (Cox ↔ AFT ↔ FHT is a config choice, not a code
fork).

The three canonical heads (CoxHead / AFTHead / FHTHead in ``survival_heads.py``)
subclass this ABC; all three delegate to the numerically-verified formulas in
``training/losses.py`` — this class adds no new math, it only unifies the interface
(behavior-preserving).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


def _losses():
    """Lazy accessor for the losses module. Deferred to call-time to avoid an
    import cycle (heads <- survival_heads <- neural_survival <- trainer <- training/
    __init__ <- ..training import). By the time a head method runs, training is fully
    initialized."""
    from ..training import losses as L
    return L


class SurvivalHead(nn.Module, ABC):
    """Abstract survival head. Concrete heads set ``model_type`` and implement
    ``forward``; ``nll`` / ``survival_at`` / ``risk`` are provided generically by
    dispatching on ``model_type`` to the verified loss formulas."""

    #: one of "cox" | "aft" | "fht" — drives the shared likelihood/survival dispatch
    model_type: str = ""

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:  # (N, k) distribution params
        ...

    # -- uniform contract --------------------------------------------------
    def nll(self, params: Tensor, time: Tensor, event: Tensor,
            reduction: str | None = None) -> Tensor:
        """Right-censored negative log-likelihood for this head's family."""
        L = _losses()
        mt = self.model_type
        if mt == "cox":
            return L.cox_ph_loss(params, time, event, reduction=reduction or "events")
        if mt == "aft":
            return L.lognormal_aft_loss(params, time, event, reduction=reduction or "mean")
        if mt == "fht":
            return L.first_hitting_time_loss(params, time, event, reduction=reduction or "mean")
        raise ValueError(f"SurvivalHead.nll: unknown model_type {mt!r}")

    def risk(self, params: Tensor, horizon: float | None = None) -> Tensor:
        """Scalar risk score per sample (higher = earlier event). Cox risk is the
        linear predictor; AFT/FHT risk is -location (verified in losses)."""
        return _losses().risk_from_output(self.model_type, params)

    def survival_at(self, params: Tensor, t: float) -> Tensor:
        """Closed-form S(t) in [0,1]. Cox requires a fitted Breslow baseline and is
        handled at eval time (raises here by design, matching losses.survival_at)."""
        return _losses().survival_at(self.model_type, params, float(t))
