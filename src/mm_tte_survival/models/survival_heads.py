"""Survival heads operating on the shared latent z: Cox / AFT / FHT.

Each head is a concrete :class:`~mm_tte_survival.models.heads.SurvivalHead`, so it
carries the uniform ``nll`` / ``survival_at`` / ``risk`` contract (dispatched by
``model_type``) in addition to ``forward``. ``forward`` behaviour is unchanged."""
from __future__ import annotations

import torch  # noqa: F401 (re-exported implicitly; kept for back-compat)
import torch.nn as nn

from .heads import SurvivalHead


class CoxHead(SurvivalHead):
    """Single risk score (log partial hazard)."""
    model_type = "cox"

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, z):
        return self.fc(z)  # (N,1)


class AFTHead(SurvivalHead):
    """Log-normal AFT: outputs (mu, log_sigma)."""
    model_type = "aft"

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 2)

    def forward(self, z):
        return self.fc(z)  # (N,2)


class FHTHead(SurvivalHead):
    """Inverse-Gaussian first-hitting-time: outputs (log_drift, log_barrier).

    ``monotone_drift`` (default) constrains log-drift to be NON-DECREASING in the latent via
    softplus weights (∂log_drift/∂z ≥ 0), so higher latent resistance ⇒ higher drift ⇒ earlier
    first passage — the same sign-identifiability the longitudinal MonotoneReadout enforces.
    The barrier stays an unconstrained affine map.
    """
    model_type = "fht"

    def __init__(self, latent_dim: int, monotone_drift: bool = True):
        super().__init__()
        self.monotone_drift = monotone_drift
        if monotone_drift:
            self.raw_w_drift = nn.Parameter(torch.zeros(latent_dim))
            self.b_drift = nn.Parameter(torch.zeros(1))
            self.fc_barrier = nn.Linear(latent_dim, 1)
        else:
            self.fc = nn.Linear(latent_dim, 2)

    def forward(self, z):
        if not self.monotone_drift:
            return self.fc(z)  # (N,2)
        w = torch.nn.functional.softplus(self.raw_w_drift)      # ≥ 0 ⇒ monotone in z
        log_drift = z @ w + self.b_drift                        # (N,)
        log_barrier = self.fc_barrier(z).squeeze(-1)            # (N,)
        return torch.stack([log_drift, log_barrier], dim=-1)    # (N,2)


_HEADS = {"cox": CoxHead, "aft": AFTHead, "fht": FHTHead}


def build_survival_heads(latent_dim: int, heads=("cox", "aft", "fht")) -> nn.ModuleDict:
    return nn.ModuleDict({h: _HEADS[h](latent_dim) for h in heads if h in _HEADS})
