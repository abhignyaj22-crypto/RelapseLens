"""Part II Stage 2 — CFM loss, exact OT coupling, and the flow→FHT bridge (toy, no real data).

Falsification built in: the trained flow must transport ρ_a→ρ_b (W2 shrinks), and a
random/frozen field must NOT — otherwise the "dynamics" carry no signal. ot_couple must match
scipy exactly. dynamic_fht_nll must reuse the verified crossing NLL and stay finite.
"""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from mm_tte_survival.models.dynamics import FlowMatchingDynamics
from mm_tte_survival.training.flow_losses import (
    conditional_flow_matching_loss, dynamic_fht_nll, ot_couple)


def _w2_proxy(x: torch.Tensor, y: torch.Tensor) -> float:
    """Exact-assignment 2-Wasserstein proxy between equal-size empirical clouds."""
    cost = (torch.cdist(x, y) ** 2).detach().cpu().numpy()
    _, col = linear_sum_assignment(cost)
    return float((((x - y[col]) ** 2).sum(-1)).mean().sqrt())


def test_ot_couple_matches_scipy():
    torch.manual_seed(0)
    a, b = torch.randn(12, 4), torch.randn(12, 4)
    got = ot_couple(a, b)
    _, col = linear_sum_assignment((torch.cdist(a, b) ** 2).numpy())
    assert torch.equal(got, b[col])


def test_cfm_transports_gaussians_random_field_does_not():
    torch.manual_seed(0)
    d, n = 4, 128
    c = torch.zeros(n, 2)  # unconditional toy

    def sample():
        return torch.randn(n, d) - 3.0, torch.randn(n, d) + 3.0

    dyn = FlowMatchingDynamics(latent_dim=d, cond_dim=2, hidden=64, n_steps=8)

    za, zb = sample()
    with torch.no_grad():
        z_end_untrained, _ = dyn.flow(za, c, torch.ones(n))
    w2_random = _w2_proxy(z_end_untrained, zb)  # untrained field ~ identity -> far from ρ_b

    opt = torch.optim.Adam(dyn.parameters(), lr=1e-2)
    loss = None
    for _ in range(400):
        za, zb = sample()
        opt.zero_grad()
        loss = conditional_flow_matching_loss(dyn, za, zb, c, sigma=0.1, ot=True)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)

    za, zb = sample()
    with torch.no_grad():
        z_end, _ = dyn.flow(za, c, torch.ones(n))
    w2_trained = _w2_proxy(z_end, zb)

    # trained field genuinely transports ρ_a → ρ_b
    assert w2_trained < 0.5 * w2_random, f"trained {w2_trained:.2f} vs random {w2_random:.2f}"
    # control: a fresh untrained/frozen field must NOT match ρ_b as well as the trained one
    frozen = FlowMatchingDynamics(latent_dim=d, cond_dim=2, hidden=64, n_steps=8)
    with torch.no_grad():
        z_frozen, _ = frozen.flow(za, c, torch.ones(n))
    assert _w2_proxy(z_frozen, zb) > w2_trained


def test_dynamic_fht_nll_is_finite_and_reuses_crossing_head():
    torch.manual_seed(0)
    d, n = 4, 16
    dyn = FlowMatchingDynamics(latent_dim=d, cond_dim=2, hidden=32)
    z0, c = torch.randn(n, d), torch.zeros(n, 2)
    w = torch.softmax(torch.randn(d), 0)      # non-negative (monotone-style) readout weights
    barrier = torch.ones(n) * 2.0
    horizon = torch.ones(n)
    time = torch.rand(n) * 10 + 0.1
    event = (torch.rand(n) > 0.5).float()
    sigma = torch.ones(n)
    loss = dynamic_fht_nll(dyn, z0, c, w, barrier, horizon, time, event, sigma)
    assert torch.isfinite(loss)
    assert loss.requires_grad  # differentiable w.r.t. the flow field
