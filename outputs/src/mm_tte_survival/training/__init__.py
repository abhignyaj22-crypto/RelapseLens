"""Training layer: losses, distillation policy, trainers."""
from .losses import (cox_ph_loss, lognormal_aft_loss, first_hitting_time_loss,
                     risk_from_output, soft_distillation_loss, d_calibration_penalty)
from .distillation import DistillationPolicy, build_distillation_policy
from .trainer import train_survival_model, FittedNeuralModel
from .trainer_legacy import fit_neural_survival, subtype_event_rate_baseline

__all__ = [
    "cox_ph_loss", "lognormal_aft_loss", "first_hitting_time_loss",
    "risk_from_output", "soft_distillation_loss", "d_calibration_penalty",
    "DistillationPolicy", "build_distillation_policy",
    "train_survival_model", "FittedNeuralModel",
    "fit_neural_survival", "subtype_event_rate_baseline",
]
