"""sota-workaround C3 — recency short-horizon capability harness design contract.

Pins the capability-axis design: near vs far time-dependent AUC, dynamic (recent + change) vs static (baseline),
triple-gated (paired ΔAUC(near) CI>0 AND BH across landmarks AND scramble-null clean). Measured verdict in
docs/sota_workaround/log.md. Real data only.
"""
from __future__ import annotations

from pathlib import Path

_AR = Path(__file__).resolve().parents[1] / "scripts" / "autoresearch"


def test_near_vs_far_auc_capability_design():
    src = (_AR / "run_recency_auc.py").read_text()
    assert "cumulative_dynamic_auc_mean" in src                      # time-dependent AUC metric
    assert "NEAR" in src and "FAR" in src                            # near-term capability vs far contrast
    # dynamic arm = most-recent + change; static = baseline
    assert 'f"last_{c}"' in src and "change_columns(LABS)" in src and 'f"base_{c}"' in src
    # triple gate
    assert 'multiplicity_adjust(pvals, method="bh")' in src
    assert 'E["ci_near"][0] > 0' in src and "p_adj < 0.05" in src and "scrambled" in src.lower()
    assert "NULL" in src and "CAPABILITY" in src


def test_reuses_shared_cohort_and_oof():
    src = (_AR / "run_recency_auc.py").read_text()
    assert "from run_landmark_trajectory import build_cohort" in src
    assert "from run_landmark_os import oof" in src
