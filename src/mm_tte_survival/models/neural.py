from __future__ import annotations

import copy
import torch
from torch import nn


class SurvivalMLP(nn.Module):
    def __init__(self, input_dim: int, model_type: str = "cox", hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.model_type = model_type.replace("opsd_", "")
        out_dim = 1 if self.model_type == "cox" else 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_ema_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(student: nn.Module, teacher: nn.Module, decay: float = 0.97) -> None:
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)
