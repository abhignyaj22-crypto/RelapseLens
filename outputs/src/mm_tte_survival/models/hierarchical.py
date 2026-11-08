"""Hierarchical Subtype Survival (HSS) — the subtype-aware model.

Shared trunk f_theta: x -> z. An agnostic head g_0 plus per-subtype heads {g_s}
over the cytogenetic set. Multi-label membership is mandatory (a patient is often
amp1q AND del17p), so the per-patient predictor is a membership-weighted mixture:

    present_i = {agnostic} ∪ {s : m_is = 1}      # agnostic always present
    w_i = softmax(gate[present] / temperature)   # learnable gate + temperature
    eta_i = Σ_k w_ik · g_k(z_i)

A patient with no called abnormality collapses to the pure agnostic head
(graceful fallback). Rare subtypes borrow strength through the shared trunk and
the agnostic mixture component. This is what the title promised — and is NOT what
the old pooled SurvivalMLP did (there, subtypes existed only in evaluation).

Cox head is the validated path (see scripts/experiments/hss_fold0_smoke.py).
AFT/FHT heads are provided structurally and require their own per-head smoke
before their IBS is reported.
"""
from __future__ import annotations

import copy
import torch
from torch import nn

_OUT_DIM = {"cox": 1, "aft": 2, "fht": 2}


class Trunk(nn.Module):
    def __init__(self, d: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())

    def forward(self, x):
        return self.net(x)


class SurvivalHead(nn.Module):
    """One survival head over the latent z. out_dim depends on the parametrisation
    (Cox:1 log-hazard; AFT/FHT:2)."""
    def __init__(self, latent_dim: int, head_type: str = "cox"):
        super().__init__()
        if head_type not in _OUT_DIM:
            raise ValueError(f"head_type must be one of {list(_OUT_DIM)}")
        self.head_type = head_type
        self.out_dim = _OUT_DIM[head_type]
        self.linear = nn.Linear(latent_dim, self.out_dim)

    def forward(self, z):
        return self.linear(z)


class MembershipMixer(nn.Module):
    """Multi-label membership-softmax over {agnostic} ∪ positive subtypes.
    Learnable per-head gate logits + a learnable softmax temperature."""
    def __init__(self, n_subtypes: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(n_subtypes + 1))   # [agnostic, sub_0..]
        self.log_temp = nn.Parameter(torch.zeros(1))

    def weights(self, m):
        present = torch.cat([torch.ones(m.shape[0], 1, device=m.device), m], dim=1)
        temp = torch.exp(self.log_temp) + 1e-3
        logits = (self.gate.unsqueeze(0) / temp).masked_fill(present < 0.5, float("-inf"))
        return torch.softmax(logits, dim=1)                     # (n, 1+n_sub)

    def forward(self, agnostic_out, subtype_outs, m):
        """agnostic_out (n,o); subtype_outs (n,s,o); m (n,s) -> mixed (n,o), w (n,1+s)."""
        w = self.weights(m)
        all_out = torch.cat([agnostic_out.unsqueeze(1), subtype_outs], dim=1)   # (n,1+s,o)
        mixed = (w.unsqueeze(-1) * all_out).sum(dim=1)
        return mixed, w


class HierarchicalSubtypeSurvival(nn.Module):
    def __init__(self, input_dim: int, n_subtypes: int, head_type: str = "cox",
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.head_type = head_type
        self.out_dim = _OUT_DIM[head_type]
        self.n_subtypes = n_subtypes
        self.trunk = Trunk(input_dim, hidden, dropout)
        self.agnostic = SurvivalHead(hidden, head_type)
        self.subs = nn.ModuleList([SurvivalHead(hidden, head_type) for _ in range(n_subtypes)])
        self.mixer = MembershipMixer(n_subtypes)

    def encode(self, x):
        return self.trunk(x)

    def head_outputs(self, z):
        ag = self.agnostic(z)
        if len(self.subs) == 0:                               # pooled / independent baseline
            subs = z.new_zeros((z.shape[0], 0, self.out_dim))
        else:
            subs = torch.stack([h(z) for h in self.subs], dim=1)
        return ag, subs

    def forward(self, x, m):
        """Returns dict: z, agnostic, subs, weights, mixed (membership-mixed output)."""
        z = self.encode(x)
        ag, subs = self.head_outputs(z)                       # (n,o), (n,s,o)
        mixed, w = self.mixer(ag, subs, m)
        return {"z": z, "agnostic": ag, "subs": subs, "weights": w, "mixed": mixed}


def make_ema_teacher(model: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_agnostic(student: HierarchicalSubtypeSurvival,
                        teacher: HierarchicalSubtypeSurvival, decay: float = 0.98) -> None:
    """EMA-update the AGNOSTIC head + trunk only (the teacher is the agnostic model).
    Ablate fixed-vs-EMA: the 2026 literature shows EMA can degrade small-n survival."""
    for tp, sp in zip(teacher.trunk.parameters(), student.trunk.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)
    for tp, sp in zip(teacher.agnostic.parameters(), student.agnostic.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)
