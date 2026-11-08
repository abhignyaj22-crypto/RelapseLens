"""Autoresearch keep/revert rule — fast unit test (no model training)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "autoresearch"))

from run_experiment import decide_keep  # noqa: E402


def _matrix(cand_uno=0.77, cand_ibs=0.125, base_ibs=0.126, rand=0.50, perm=0.70):
    return pd.DataFrame([
        {"arm": "clinical+omics", "mean_uno_cindex": 0.751, "mean_ipcw_ibs": base_ibs},
        {"arm": "cand", "mean_uno_cindex": cand_uno, "mean_ipcw_ibs": cand_ibs},
        {"arm": "permuted_omics", "mean_uno_cindex": perm, "mean_ipcw_ibs": 0.14},
        {"arm": "random_label", "mean_uno_cindex": rand, "mean_ipcw_ibs": 0.16},
    ])


def _deltas(delta=0.02, ci_low=0.005, sig=True):
    return pd.DataFrame([{"comparison": "cand_vs_clinical+omics", "delta_cindex": delta,
                          "delta_ci_low": ci_low, "fdr_significant": sig}])


def test_keep_when_all_criteria_met():
    keep, s, _ = decide_keep(_matrix(), _deltas(), "cand", "clinical+omics")
    assert keep and all(s["reasons"].values())


def test_revert_when_ci_crosses_zero():
    keep, s, _ = decide_keep(_matrix(), _deltas(ci_low=-0.001), "cand", "clinical+omics")
    assert not keep and not s["reasons"]["ci_low_gt_0"]


def test_revert_when_not_bh_significant():
    keep, _, _ = decide_keep(_matrix(), _deltas(sig=False), "cand", "clinical+omics")
    assert not keep


def test_revert_when_ibs_regresses():
    keep, s, _ = decide_keep(_matrix(cand_ibs=0.140), _deltas(), "cand", "clinical+omics")
    assert not keep and not s["reasons"]["ibs_not_worse"]


def test_revert_when_random_label_leaks():
    keep, s, _ = decide_keep(_matrix(rand=0.72), _deltas(), "cand", "clinical+omics")
    assert not keep and not s["reasons"]["sanity_random_label"]


def test_revert_when_not_beating_permuted_omics():
    keep, s, _ = decide_keep(_matrix(cand_uno=0.69, perm=0.70), _deltas(), "cand", "clinical+omics")
    assert not keep and not s["reasons"]["beats_permuted_omics"]
