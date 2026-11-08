"""Cross-framework integration seam (merge dedup D5/D7/D8/D9).

The single place where the ResistanceMap/MORT-FM platform reaches the trusted survival/eval
core (`mm_tte_survival`). One-way dependency only. See ``core_bridge``.
"""
from resistancemap.integration.core_bridge import (
    FUSION_REGISTRY,
    build_fusion,
    score_survival_ipcw,
    harrell_c,
    score_hitting_times,
    honest_claim_report,
)

__all__ = [
    "FUSION_REGISTRY",
    "build_fusion",
    "score_survival_ipcw",
    "harrell_c",
    "score_hitting_times",
    "honest_claim_report",
]
