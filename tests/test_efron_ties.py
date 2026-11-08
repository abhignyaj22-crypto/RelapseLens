"""Quick-win: Efron tie correction in cox_ph_loss (matches lifelines' default).

Efron must reduce EXACTLY to Breslow when there are no tied event times, must match a
hand-computed value under ties, and must differ from Breslow when ties are present.
"""
from __future__ import annotations

import math

import torch

from mm_tte_survival.training.losses import cox_ph_loss


def test_efron_equals_breslow_without_ties():
    risk = torch.tensor([0.2, -0.3, 0.5, -0.1])
    time = torch.tensor([5.0, 3.0, 4.0, 2.0])          # all distinct
    event = torch.tensor([1.0, 1.0, 0.0, 1.0])
    b = cox_ph_loss(risk, time, event, reduction="sum", ties="breslow")
    e = cox_ph_loss(risk, time, event, reduction="sum", ties="efron")
    assert torch.allclose(b, e, atol=1e-5)


def test_efron_matches_manual_with_ties():
    risk = torch.tensor([0.5, -0.5, 0.0])
    time = torch.tensor([1.0, 1.0, 2.0])               # two deaths tied at t=1
    event = torch.tensor([1.0, 1.0, 0.0])
    got = cox_ph_loss(risk, time, event, reduction="sum", ties="efron")
    e0, e1, e2 = (math.exp(x) for x in (0.5, -0.5, 0.0))
    S_risk, S_death = e0 + e1 + e2, e0 + e1            # risk set = all; tied deaths = {0,1}
    ll = (0.5 + -0.5) - (math.log(S_risk) + math.log(S_risk - 0.5 * S_death))
    assert torch.allclose(got, torch.tensor(-ll), atol=1e-5)


def test_efron_differs_from_breslow_with_ties():
    risk = torch.tensor([0.5, -0.5, 0.0, 0.3])
    time = torch.tensor([1.0, 1.0, 2.0, 1.0])          # 3 tied deaths at t=1
    event = torch.tensor([1.0, 1.0, 0.0, 1.0])
    b = cox_ph_loss(risk, time, event, reduction="sum", ties="breslow")
    e = cox_ph_loss(risk, time, event, reduction="sum", ties="efron")
    assert not torch.allclose(b, e, atol=1e-3)


def test_efron_is_differentiable():
    risk = torch.zeros(6, requires_grad=True)
    time = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    event = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    loss = cox_ph_loss(risk, time, event, ties="efron")
    loss.backward()
    assert risk.grad is not None and torch.isfinite(risk.grad).all()
