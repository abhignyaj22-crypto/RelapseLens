"""Regression test: OPSD self-distillation must actually change the trained model.

Previously `train_survival_model` accepted a DistillationPolicy and ignored it, so
`cox` and `opsd_cox` produced byte-identical models. This pins the fix: the L2
divergence of the trained weights from the no-distillation solution grows with the
distillation weight (and is exactly zero when the policy is disabled).
"""
import copy
import numpy as np
import torch

from mm_tte_survival.models.neural_survival import build_multihead_model
from mm_tte_survival.training.distillation import DistillationPolicy
from mm_tte_survival.training.trainer import train_survival_model


def _toy(seed=0, n=120, p=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype("float32")
    risk = X[:, 0] * 0.8 - X[:, 1] * 0.5
    t = (np.exp(-risk) * rng.weibull(1.3, n) * 20 + 1).astype("float32")
    e = (rng.random(n) < 0.6).astype("float32")
    return X, t, e


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _train_with_weight(w, base_model, X, t, e):
    model = copy.deepcopy(base_model)                 # identical initialization each run
    torch.manual_seed(0)
    pol = DistillationPolicy(enabled=(w > 0), start_epoch=0, ramp_epochs=0,
                             weight=w, ema_decay=0.9)
    fit = train_survival_model(X, t, e, model, [f"x{i}" for i in range(X.shape[1])],
                               epochs=80, lr=1e-2, distillation=pol)
    return _flat_params(fit.model), fit


def test_distillation_weight_changes_the_model_monotonically():
    X, t, e = _toy()
    torch.manual_seed(0)
    base = build_multihead_model(X.shape[1], {"latent_dim": 16, "hidden_dims": [16]}, heads=("cox",))

    ref, ref_fit = _train_with_weight(0.0, base, X, t, e)
    assert all(w == 0.0 for w in ref_fit.distill_weight_history)   # disabled -> no distill weight

    divergences = []
    for w in (0.6, 3.0):
        params, fit = _train_with_weight(w, base, X, t, e)
        assert max(fit.distill_weight_history) > 0.0                # OPSD actually engaged
        divergences.append(float(torch.linalg.norm(params - ref)))

    # w=0 reproduces the reference exactly; larger distillation weight => larger change
    zero_again, _ = _train_with_weight(0.0, base, X, t, e)
    assert float(torch.linalg.norm(zero_again - ref)) < 1e-6
    assert divergences[0] > 1e-4
    assert divergences[1] > divergences[0]
