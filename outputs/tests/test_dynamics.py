"""Part II Stage 1 — LatentDynamics abstraction + the backward-compatibility anchor.

The load-bearing guarantee: adding the dynamics seam must not change today's model.
``StaticDynamics`` (v≡0) makes ``ResistanceModel`` reproduce its static forward pass
bit-for-bit, and the dynamic branch only activates when a real dynamics AND a horizon are
supplied. FlowMatchingDynamics is only shape/identity-checked here; its learning is Stage 2.
"""
from __future__ import annotations

import torch

from mm_tte_survival.models.dynamics import (
    FlowMatchingDynamics, LatentDynamics, StaticDynamics)
from mm_tte_survival.models.encoders import TranscriptomeEncoder
from mm_tte_survival.models.fusion import ResidualFusion
from mm_tte_survival.models.resistance_model import ResistanceModel
from mm_tte_survival.models.survival_heads import CoxHead

D = 16


def _model(dynamics=None):
    torch.manual_seed(0)
    enc = {"clinical": TranscriptomeEncoder(6, latent_dim=D),
           "transcriptomics": TranscriptomeEncoder(20, latent_dim=D)}
    dims = {"clinical": D, "transcriptomics": D}
    return ResistanceModel(enc, ResidualFusion(dims, out_dim=D), CoxHead(D), dynamics=dynamics)


def _batch(n=8):
    torch.manual_seed(1)
    return {"clinical": torch.randn(n, 6), "transcriptomics": torch.randn(n, 20)}


def test_static_dynamics_velocity_and_flow_are_identity():
    d = StaticDynamics(D)
    z, c, tau = torch.randn(5, D), torch.randn(5, 8), torch.rand(5)
    assert torch.equal(d.velocity(z, c, tau), torch.zeros_like(z))
    z_out, v_mean = d.flow(z, c, tau)
    assert torch.equal(z_out, z)
    assert torch.equal(v_mean, torch.zeros_like(z))


def test_static_dynamics_is_a_latent_dynamics():
    assert isinstance(StaticDynamics(4), LatentDynamics)
    assert isinstance(FlowMatchingDynamics(4, 2), LatentDynamics)


def test_resistance_model_static_is_bit_for_bit_identity():
    """dynamics=None (today) and dynamics=StaticDynamics must give identical outputs."""
    m = _model(dynamics=None)
    m.eval()
    batch = _batch()
    with torch.no_grad():
        out_none = m(batch)
        m.dynamics = StaticDynamics(D)          # same weights, just add the anchor
        out_static = m(batch)                   # no horizon_tau -> static path
    for k in ("z", "params", "risk"):
        assert torch.equal(out_none[k], out_static[k]), f"{k} diverged under StaticDynamics"
    assert out_none["model_type"] == out_static["model_type"] == "cox"


def test_static_dynamics_ignores_horizon():
    """Even WITH a horizon supplied, StaticDynamics keeps the static (head) path."""
    m = _model(dynamics=StaticDynamics(D))
    m.eval()
    batch = _batch()
    with torch.no_grad():
        ref = _model(dynamics=None).eval()(batch)
        out = m(batch, c=torch.randn(8, 8), horizon_tau=torch.rand(8))
    assert torch.equal(ref["params"], out["params"])
    assert "z_tau" not in out and "params" in out


def test_flow_matching_shapes_and_zero_horizon_identity():
    torch.manual_seed(0)
    d = FlowMatchingDynamics(latent_dim=D, cond_dim=8)
    z0, c = torch.randn(4, D), torch.randn(4, 8)
    v = d.velocity(z0, c, torch.rand(4))
    assert v.shape == (4, D)
    z_at_0, _ = d.flow(z0, c, torch.zeros(4))       # tau_max=0 -> no displacement
    assert torch.allclose(z_at_0, z0, atol=1e-6)
    z_at_1, v_mean = d.flow(z0, c, torch.ones(4))
    assert z_at_1.shape == (4, D) and v_mean.shape == (4, D)


def test_dynamic_branch_activates_with_flow_dynamics():
    """A real dynamics + horizon returns the flowed latent, not head params."""
    m = _model(dynamics=FlowMatchingDynamics(latent_dim=D, cond_dim=8))
    m.eval()
    batch = _batch()
    with torch.no_grad():
        out = m(batch, c=torch.randn(8, 8), horizon_tau=torch.rand(8))
    assert set(out) == {"z0", "z_tau", "v_mean"}
    assert out["z_tau"].shape == (8, D)
