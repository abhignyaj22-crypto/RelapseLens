"""Trainer for the abstract MultiHeadSurvivalModel (optional neural path).

The default production model is ResidualRiskModel (see models/residual_risk.py);
this trains the parallel neural encoder + Cox head when a run explicitly enables
it. Kept deliberately small — the legacy per-model trainer in trainer_legacy.py
is used by `run-experiments`.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from ..models.neural_survival import MultiHeadSurvivalModel
from ..models.neural import make_ema_teacher, update_ema_teacher
from ..survival_curves import breslow_baseline, time_grid
from .losses import cox_ph_loss, cox_survival_curve, soft_distillation_loss, survival_curve_distill
from .distillation import DistillationPolicy


@dataclass
class FittedNeuralModel:
    model: MultiHeadSurvivalModel
    feature_names: list[str]
    history: list[float]
    distill_weight_history: list[float] | None = None


def train_survival_model(X_train, t_train, e_train, model: MultiHeadSurvivalModel,
                         feature_names, epochs=100, lr=1e-3, weight_decay=1e-4,
                         distillation: DistillationPolicy | None = None) -> FittedNeuralModel:
    """Train the neural Cox head, optionally with on-policy self-distillation (OPSD).

    When `distillation.enabled`, an EMA teacher (a decayed copy of the student) is
    maintained and the loss gains a scheduled consistency term
    `w(epoch) * MSE(r_student, r_teacher)` (temperature-free standardized-risk MSE,
    see losses.soft_distillation_loss and docs/model_math.md). w(epoch) follows the
    policy's warmup+ramp; the teacher is EMA-updated after every optimizer step:
    phi <- decay*phi + (1-decay)*theta. This makes OPSD an ACTIVE regularizer rather
    than the former silent no-op.
    """
    Xt = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    tt = torch.tensor(np.asarray(t_train), dtype=torch.float32)
    et = torch.tensor(np.asarray(e_train), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_distill = distillation is not None and getattr(distillation, "enabled", False)
    teacher = make_ema_teacher(model) if use_distill else None
    # CALIBRATED OPSD: distill the teacher's survival CURVE (via a Breslow baseline), not just
    # the risk ranking. Falls back to ranking MSE only when no time grid is available (tiny cohort).
    grid = time_grid(tt.numpy(), et.numpy(), tt.numpy()) if use_distill else None
    history, w_hist = [], []
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        risk = out["cox"].squeeze(-1)
        loss = cox_ph_loss(risk, tt, et, ties="efron")   # matches lifelines headline; correct under ties
        w = distillation.weight_at(epoch) if use_distill else 0.0
        if w > 0.0:
            with torch.no_grad():
                teacher_risk = teacher(Xt)["cox"].squeeze(-1)
            if grid is not None:
                with torch.no_grad():
                    H = torch.tensor(breslow_baseline(tt.numpy(), et.numpy(),
                                                      teacher_risk.numpy(), grid), dtype=torch.float32)
                s_student = cox_survival_curve(risk, H)
                s_teacher = cox_survival_curve(teacher_risk, H)
                loss = loss + w * survival_curve_distill(s_student, s_teacher)
            else:
                loss = loss + w * soft_distillation_loss(risk, teacher_risk)
        loss.backward()
        opt.step()
        if use_distill:
            update_ema_teacher(model, teacher, decay=distillation.ema_decay)
        history.append(float(loss.item()))
        w_hist.append(float(w))
    return FittedNeuralModel(model=model, feature_names=list(feature_names),
                             history=history, distill_weight_history=w_hist)
