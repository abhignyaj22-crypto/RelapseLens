"""core_bridge — the single cross-framework seam (merge dedup D5/D7/D8/D9).

After merging the ResistanceMap/MORT-FM platform into the mm_tte_opisd repo, this module is
the ONE connective tissue through which the platform's diverse models reach the trusted
survival/eval core (`mm_tte_survival`). It enforces the merge's core principle — the trusted
core is the single scoring/honesty authority — without rewiring the platform's tested pipelines.

Provides:
  * build_fusion / FUSION_REGISTRY  (D5) — one registry of multimodal-fusion modules: RM's
    MultiOmicFoundationFusion is the PRIMARY architecture; the core's CrossAttentionFusion /
    LateConcatFusion / ResidualFusion are the BASELINE arms.
  * score_survival_ipcw / harrell_c — the single survival-discrimination authority (core Uno
    IPCW + Harrell). ANY platform model's risk scores (D7 PPI-GNN propagation, D8 drug-
    conditioned head, LENS, foundation head) are scored here, on the same axis as the core.
  * score_hitting_times (D9) — bridge from RM's graph-energy-SDE first-passage statistics
    (HittingTime.mean_tau / frac_hit) to core IPCW concordance.
  * honest_claim_report — the single claim-honesty authority (core build_claim_report), so a
    platform claim (e.g. a drug-response or PPI result) is gated by the same rules as the core.

One-way dependency only: this module imports mm_tte_survival; the core never imports back.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# D5 — unified fusion registry (RM foundation = primary; core = baseline arms)
# --------------------------------------------------------------------------- #
def _make_cross_attention(**kw):
    from mm_tte_survival.models.fusion import CrossAttentionFusion
    return CrossAttentionFusion(**kw)


def _make_late_concat(**kw):
    from mm_tte_survival.models.fusion import LateConcatFusion
    return LateConcatFusion(**kw)


def _make_residual(**kw):
    from mm_tte_survival.models.fusion import ResidualFusion
    return ResidualFusion(**kw)


# name -> (constructor, role). Honest-core edition: the RM/MORT-FM breadth fusions (foundation, cross_modal,
# tensor, causal, …) and the VAE/ODE/SDE generative+dynamics registries were removed with the breadth. The
# registry now holds the trusted core's fusion baseline arms — the only fusions this repo actually implements.
FUSION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cross_attention": {"factory": _make_cross_attention, "role": "baseline",
                        "impl": "mm_tte_survival.models.fusion.CrossAttentionFusion"},
    "late_concat": {"factory": _make_late_concat, "role": "baseline",
                    "impl": "mm_tte_survival.models.fusion.LateConcatFusion"},
    "residual": {"factory": _make_residual, "role": "baseline",
                 "impl": "mm_tte_survival.models.fusion.ResidualFusion"},
}


def build_fusion(name: str, **kwargs):
    """Construct a multimodal-fusion module by registry name — the ONE canonical fusion accessor.

    name in {cross_attention, late_concat, residual} -> core baseline fusions (take modality_dims).
    kwargs are forwarded verbatim to the underlying class constructor.
    """
    if name not in FUSION_REGISTRY:
        raise KeyError(f"unknown fusion '{name}'; choices: {sorted(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[name]["factory"](**kwargs)


# --------------------------------------------------------------------------- #
# Single survival-scoring authority (used by D7 PPI, D8 drug, and any RM risk)
# --------------------------------------------------------------------------- #
def score_survival_ipcw(t_train, e_train, t_test, e_test, risk_test, tau: Optional[float] = None) -> float:
    """Uno IPCW C-index via the trusted core (higher risk = earlier event). NaN if undefined.
    This is the single discrimination axis for every platform model's risk scores."""
    from mm_tte_survival.metrics_ipcw import uno_c_index
    return uno_c_index(t_train, e_train, t_test, e_test, np.asarray(risk_test, float), tau=tau)


def harrell_c(time, event, risk) -> float:
    """Harrell C-index via the trusted core (single canonical implementation)."""
    from mm_tte_survival.metrics import fast_c_index
    return float(fast_c_index(time, event, risk))


# --------------------------------------------------------------------------- #
# D9 — graph-energy-SDE first-passage -> core IPCW concordance
# --------------------------------------------------------------------------- #
def score_hitting_times(t_train, e_train, t_test, e_test, *,
                        mean_tau=None, frac_hit=None, tau: Optional[float] = None) -> float:
    """Score RM's SDE first-passage statistics against observed OS with core Uno IPCW.

    Consumes the outputs of ``resistancemap.mortfm.trajectory.hitting_time.HittingTime.forward``:
      * ``frac_hit`` (B,)  — P(reach resistant basin); higher = higher risk (preferred proxy).
      * ``mean_tau`` (B,)  — E[first-passage time | hit]; EARLIER = higher risk (used if
        frac_hit is absent; never-hit NaNs map to the lowest risk).
    Returns the Uno IPCW C-index of that risk proxy vs (t_test, e_test)."""
    if frac_hit is not None:
        risk = np.asarray(frac_hit, float)
    elif mean_tau is not None:
        mt = np.asarray(mean_tau, float)
        # earlier passage => higher risk; never-hit (NaN) => lowest risk (most negative risk)
        filled = np.nan_to_num(mt, nan=np.nanmax(mt) + 1.0 if np.isfinite(mt).any() else 0.0)
        risk = -filled
    else:
        raise ValueError("provide frac_hit or mean_tau from HittingTime.forward()")
    return score_survival_ipcw(t_train, e_train, t_test, e_test, risk, tau=tau)


# --------------------------------------------------------------------------- #
# Single claim-honesty authority (gates D7/D8 platform claims like the core)
# --------------------------------------------------------------------------- #
def honest_claim_report(endpoint_spec: dict, **kwargs) -> dict:
    """Build a claim report via the trusted core's ``build_claim_report`` so a platform result
    (drug response, PPI, foundation) is gated by exactly the same honesty rules as the core."""
    from mm_tte_survival.evaluation.claim_gate import build_claim_report
    return build_claim_report(endpoint_spec, **kwargs)
