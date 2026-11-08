"""Quick-win: monotone drift in the neural FHTHead (sign identifiability).

With monotone_drift=True the log-drift is non-decreasing in every latent coordinate (softplus
weights), so a uniformly larger latent cannot yield a smaller drift. The legacy unconstrained
head remains available for backward comparison.
"""
from __future__ import annotations

import torch

from mm_tte_survival.models.survival_heads import FHTHead


def test_fht_drift_is_monotone_in_latent():
    torch.manual_seed(0)
    head = FHTHead(8, monotone_drift=True)
    z1 = torch.randn(16, 8)
    z2 = z1 + torch.rand(16, 8)                 # z2 >= z1 elementwise
    ld1 = head(z1)[:, 0]                         # log_drift
    ld2 = head(z2)[:, 0]
    assert (ld2 >= ld1 - 1e-6).all(), "log-drift decreased under a larger latent"
    assert head(z1).shape == (16, 2)


def test_fht_monotone_default_and_legacy_shape():
    assert FHTHead(6).monotone_drift is True                     # monotone by default
    legacy = FHTHead(6, monotone_drift=False)
    assert hasattr(legacy, "fc") and legacy(torch.randn(4, 6)).shape == (4, 2)


def test_fht_head_nll_still_finite():
    from mm_tte_survival.models.survival_heads import FHTHead as H
    head = H(8)
    z = torch.randn(20, 8)
    params = head(z)
    t = torch.rand(20) * 10 + 0.1
    e = (torch.rand(20) > 0.5).float()
    loss = head.nll(params, t, e)
    assert torch.isfinite(loss)
