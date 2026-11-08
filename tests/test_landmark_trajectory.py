"""sota-workaround C1 — landmark-trajectory harness design contract (pins rigor, not the ~3min run).

The trajectory-vs-static comparison must be triple-gated: paired bootstrap td-C CI>0 AND BH-adjusted
permutation significance across all comparisons AND a change-feature permutation null that collapses the gain.
Measured verdict: NULL — no comparison clears all three (12-mo Uno-CI>0 [+0.004,+0.132] but BH-p=0.21 and the
change features add ~0; the modest gain is a most-recent-level recency effect, not a trajectory effect). See
docs/sota_workaround/log.md. Real data only.
"""
from __future__ import annotations

from pathlib import Path

_AR = Path(__file__).resolve().parents[1] / "scripts" / "autoresearch"


def test_triple_gate_and_reused_rigor():
    src = (_AR / "run_landmark_trajectory.py").read_text()
    # reuses the shared rigor layer (no reimplementation)
    assert "from mm_tte_survival.evaluation.stats import paired_permutation_test, multiplicity_adjust" in src
    assert "from run_landmark_os import" in src                       # same at-risk cohort / OOF scaffolding
    # triple gate: CI>0 AND BH-significant AND permutation-null clean
    assert 'multiplicity_adjust(pvals, method="bh")' in src
    assert 'ent["ci"][0] > 0' in src and "p_adj < 0.05" in src
    assert "trajectory_minus_permuted_change" in src and "clean" in src
    assert "NULL" in src and "CAPABILITY" in src                      # both honest paths exist


def test_arms_are_static_locf_trajectory():
    src = (_AR / "run_landmark_trajectory.py").read_text()
    assert "arm_columns" in src and "change_columns" in src
