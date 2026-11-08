"""On-policy self-distillation (OPSD) policy — optional and claim-gated.

OPSD is retained only when it improves calibration, repeated-split stability, or
paired ΔC; the policy object simply schedules the distillation weight so the
trainer/evaluator can decide whether to keep it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DistillationPolicy:
    enabled: bool = False
    start_epoch: int = 40
    ramp_epochs: int = 20
    weight: float = 0.05
    ema_decay: float = 0.98
    temperature: float = 2.0

    def weight_at(self, epoch: int) -> float:
        if not self.enabled or epoch < self.start_epoch:
            return 0.0
        if self.ramp_epochs <= 0:
            return self.weight
        frac = min(1.0, (epoch - self.start_epoch) / self.ramp_epochs)
        return self.weight * frac


def build_distillation_policy(enabled=False, start_epoch=40, ramp_epochs=20,
                              weight=0.05, ema_decay=0.98, temperature=2.0) -> DistillationPolicy:
    return DistillationPolicy(enabled=enabled, start_epoch=start_epoch,
                              ramp_epochs=ramp_epochs, weight=weight,
                              ema_decay=ema_decay, temperature=temperature)
