"""Part II Stage 4 (correctness guarantees) — the flow→FHT coupling degrades safely.

Two guarantees that need no training:
  * with StaticDynamics the dynamic FHT NLL equals the zero-drift first passage, i.e. it
    reduces EXACTLY to the static (verified) crossing NLL — the compatibility anchor holds
    through the FHT coupling, not just the forward pass;
  * the crossing survival curve S(t) from the head is monotone non-increasing in t (a valid
    survival function), so any per-patient dynamic survival curve is monotone by construction.
The empirical "does the flow beat static" test is Stage 5 (needs real training).
"""
from __future__ import annotations

import torch

from mm_tte_survival.longitudinal.crossing_head import crossing_log_survival, crossing_nll
from mm_tte_survival.models.dynamics import StaticDynamics
from mm_tte_survival.training.flow_losses import dynamic_fht_nll


def test_dynamic_fht_reduces_to_static_under_static_dynamics():
    torch.manual_seed(0)
    d, n = 8, 32
    dyn = StaticDynamics(d)
    z0 = torch.randn(n, d)
    c = torch.zeros(n, 2)
    w = torch.softmax(torch.randn(d), 0)
    barrier = torch.ones(n) * 2.0
    horizon = torch.rand(n) + 0.5
    time = torch.rand(n) * 10 + 0.1
    event = (torch.rand(n) > 0.5).float()
    sigma = torch.ones(n)

    got = dynamic_fht_nll(dyn, z0, c, w, barrier, horizon, time, event, sigma)
    # static dynamics => v_mean == 0 => drift == 0; the coupling must equal the zero-drift FHT
    gap = (barrier - (z0 * w).sum(-1)).clamp_min(1e-6)
    ref = crossing_nll(gap, torch.zeros(n), sigma, time, event).mean()
    assert torch.allclose(got, ref, atol=1e-6)


def test_crossing_survival_is_monotone_nonincreasing():
    torch.manual_seed(0)
    n = 16
    gap = torch.rand(n) + 0.5
    drift = torch.randn(n) * 0.2          # mixed drift signs (progressors + responders)
    sigma = torch.ones(n)
    ts = torch.linspace(0.1, 50.0, 40)
    prev = torch.full((n,), 0.0)          # log S(0-) ~ 0
    for t in ts:
        log_s = crossing_log_survival(gap, drift, sigma, torch.full((n,), float(t)))
        assert torch.isfinite(log_s).all()
        assert (log_s <= prev + 1e-5).all(), "survival increased with time — not a valid S(t)"
        prev = log_s
