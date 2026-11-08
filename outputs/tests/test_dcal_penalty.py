"""Quick-win: differentiable D-calibration / smoothed-ECE training penalty.

Uniform event survival-probabilities (calibrated) give a near-zero penalty; clustered ones
(mis-calibrated) give a strictly larger penalty; the penalty is differentiable.
"""
from __future__ import annotations

import torch

from mm_tte_survival.training.losses import d_calibration_penalty


def test_uniform_is_low_clustered_is_high():
    torch.manual_seed(0)
    n = 400
    event = torch.ones(n)
    s_uniform = torch.rand(n)                       # calibrated: S(t_i) ~ Uniform(0,1)
    s_clustered = torch.full((n,), 0.5) + 0.02 * torch.randn(n)  # all piled in one bin
    p_uniform = d_calibration_penalty(s_uniform, event)
    p_clustered = d_calibration_penalty(s_clustered, event)
    assert p_uniform < p_clustered
    assert p_uniform < 5e-3                          # near-uniform -> near-zero


def test_penalty_is_differentiable():
    s = torch.rand(50, requires_grad=True)
    event = (torch.rand(50) > 0.3).float()
    pen = d_calibration_penalty(s, event)
    pen.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_too_few_events_is_zero():
    s = torch.rand(5)
    event = torch.tensor([1.0, 0, 0, 0, 0])          # one event -> undefined -> zero
    assert float(d_calibration_penalty(s, event)) == 0.0
