"""ABC / Liskov-substitution contract tests for the five OOP base classes.

These lock the polymorphic contract introduced in the ResistanceMap × MMTTE refactor:
any SurvivalHead is substitutable in the loss/eval path, any ModalityEncoder returns a
(N, d_latent) latent, any FusionModule maps a dict of latents to a fused tensor, the
composed ResistanceModel runs end-to-end, DataSource rejects controlled tiers, and the
canonical interpretable ResidualRiskModel is still reachable + additive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mm_tte_survival.models.heads import SurvivalHead
from mm_tte_survival.models.survival_heads import CoxHead, AFTHead, FHTHead
from mm_tte_survival.models.encoders import (
    ModalityEncoder, TranscriptomeEncoder, ProgramActivityEncoder,
    VAEEncoder, PPIGraphEncoder, ScRNAStateEncoder,
)
from mm_tte_survival.models.fusion import (
    FusionModule, LateConcatFusion, CrossAttentionFusion, ResidualFusion,
)
from mm_tte_survival.models.resistance_model import ResistanceModel
from mm_tte_survival.models.residual_risk import ResidualRiskModel
from mm_tte_survival.data import sources as S


# ---- SurvivalHead --------------------------------------------------------
ALL_HEADS = [CoxHead, AFTHead, FHTHead]


@pytest.mark.parametrize("HeadCls", ALL_HEADS)
def test_survival_head_is_liskov_substitutable(HeadCls):
    head = HeadCls(latent_dim=8)
    assert isinstance(head, SurvivalHead)
    assert head.model_type in {"cox", "aft", "fht"}
    z = torch.randn(16, 8)
    params = head(z)
    assert params.shape[0] == 16
    time = torch.rand(16) * 10 + 0.1
    event = (torch.rand(16) > 0.5).float()
    loss = head.nll(params, time, event)
    assert loss.ndim == 0 and torch.isfinite(loss)
    risk = head.risk(params)
    assert risk.reshape(-1).shape[0] == 16


@pytest.mark.parametrize("HeadCls", [AFTHead, FHTHead])
def test_survival_head_survival_at_is_probability(HeadCls):
    head = HeadCls(latent_dim=8)
    s = head.survival_at(head(torch.randn(16, 8)), t=12.0)
    assert torch.all((s >= 0) & (s <= 1))


def test_cox_survival_at_requires_baseline():
    with pytest.raises(ValueError):
        CoxHead(8).survival_at(CoxHead(8)(torch.randn(4, 8)), t=12.0)


# ---- ModalityEncoder -----------------------------------------------------
def test_modality_encoders_return_shared_latent():
    encs = [TranscriptomeEncoder(20, latent_dim=16),
            ProgramActivityEncoder(10, latent_dim=16),
            ScRNAStateEncoder(30, latent_dim=16)]
    x = {20: torch.randn(8, 20), 10: torch.randn(8, 10), 30: torch.randn(8, 30)}
    for enc, dim in zip(encs, [20, 10, 30]):
        assert isinstance(enc, ModalityEncoder)
        out = enc(x[dim])
        assert out.shape == (8, 16)


def test_stage_d_encoders_declare_latent_dim():
    # VAE/PPI encoders exist (Stage-D bodies) and expose the interface
    for E in (VAEEncoder, PPIGraphEncoder):
        e = E(input_dim=12, latent_dim=16)
        assert isinstance(e, ModalityEncoder)
        assert e.latent_dim == 16


# ---- FusionModule --------------------------------------------------------
def _latents(dims, n=8):
    return {name: torch.randn(n, d) for name, d in dims.items()}


def test_fusion_modules_map_dict_to_tensor():
    dims = {"clinical": 8, "transcriptomics": 16, "programs": 8}
    fusions = [
        LateConcatFusion(dims, out_dim=16),
        CrossAttentionFusion(dims, d_model=16),
        ResidualFusion(dims, clinical_key="clinical", out_dim=16),
    ]
    lat = _latents(dims)
    for f in fusions:
        assert isinstance(f, FusionModule)
        out = f(lat)
        assert out.shape[0] == 8 and out.ndim == 2


def test_fusion_handles_missing_modality_block():
    dims = {"clinical": 8, "transcriptomics": 16, "programs": 8}
    lat = _latents(dims)
    del lat["programs"]  # a block absent at runtime
    # concat pads, attention skips, residual omits — none should raise
    assert LateConcatFusion(dims, out_dim=16)(lat).shape == (8, 16)
    assert ResidualFusion(dims, out_dim=16)(lat).shape == (8, 16)


def test_residual_fusion_requires_clinical():
    with pytest.raises(ValueError):
        ResidualFusion({"transcriptomics": 16}, clinical_key="clinical")


# ---- ResistanceModel composition ----------------------------------------
def test_resistance_model_composes_end_to_end():
    encoders = {"clinical": ClinicalLatent(6, 16),
                "transcriptomics": TranscriptomeEncoder(20, latent_dim=16)}
    dims = {"clinical": 16, "transcriptomics": 16}
    model = ResistanceModel(encoders, ResidualFusion(dims, out_dim=16), CoxHead(16))
    batch = {"clinical": torch.randn(8, 6), "transcriptomics": torch.randn(8, 20)}
    out = model(batch)
    assert out["params"].shape[0] == 8
    assert out["model_type"] == "cox"
    loss = model.nll(batch, torch.rand(8) * 5 + 0.1, (torch.rand(8) > 0.5).float())
    assert torch.isfinite(loss)


class ClinicalLatent(TranscriptomeEncoder):
    """tiny clinical encoder for the composition test (reuses MLP encoder)."""


def test_resistance_model_swaps_head_by_construction():
    dims = {"clinical": 16}
    enc = {"clinical": TranscriptomeEncoder(6, latent_dim=16)}
    batch = {"clinical": torch.randn(4, 6)}
    for Head in ALL_HEADS:
        m = ResistanceModel(enc, ResidualFusion(dims, out_dim=16), Head(16))
        assert m(batch)["model_type"] == Head(16).model_type


# ---- canonical interpretable path still intact ---------------------------
def test_residual_risk_model_reachable_and_additive():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.normal(size=60), "iss": rng.normal(size=60),
        "mol1": rng.normal(size=60), "mol2": rng.normal(size=60),
        "time_months": rng.uniform(1, 60, size=60),
        "event": rng.integers(0, 2, size=60),
    })
    m = ResistanceModel.interpretable_residual(["age", "iss"], ["mol1", "mol2"])
    assert isinstance(m, ResidualRiskModel)
    pred = m.fit(df).predict(df)
    # exact additive identity: total == clinical + molecular_residual
    np.testing.assert_allclose(
        pred["total_risk"].values,
        (pred["clinical_risk"] + pred["molecular_residual_risk"]).values,
        rtol=1e-9, atol=1e-9,
    )


# ---- DataSource controlled-access rejection ------------------------------
def test_datasource_rejects_controlled_ids():
    for cid in ["commpass_pfs_controlled", "dbgap_raw", "beataml_raw_x"]:
        with pytest.raises(ValueError):
            S.assert_public(cid)


def test_public_sources_registered():
    assert "string_v12_9606" in S.PUBLIC_SOURCES
    assert "ccle_proteomics_gygi" in S.PUBLIC_SOURCES
