"""Tests for Stage-D negative-control mechanics."""
import numpy as np
import pandas as pd
import pytest

from mm_tte_survival.experiments_stageD import _membership, _decide


def test_permuted_preserves_prevalence_and_cooccurrence():
    rng = np.random.default_rng(0)
    M = (rng.random((200, 5)) < [0.1, 0.2, 0.3, 0.4, 0.5]).astype("float32")
    P = _membership(M, "permuted", rng)
    # column sums (prevalence) identical; it is a row permutation
    assert np.array_equal(M.sum(axis=0), P.sum(axis=0))
    # co-occurrence structure (a patient's label vector) is preserved as a set of rows
    assert sorted(map(tuple, M.tolist())) == sorted(map(tuple, P.tolist()))


def test_random_matches_marginal_prevalence_only():
    rng = np.random.default_rng(1)
    M = (rng.random((4000, 3)) < [0.1, 0.3, 0.6]).astype("float32")
    R = _membership(M, "random", rng)
    assert np.allclose(R.mean(axis=0), M.mean(axis=0), atol=0.03)


def test_decision_stops_when_real_matches_controls():
    # real improvement ~ permuted/random -> REGULARIZATION/STOP
    sub = ["del17p", "del1p", "amp1q"]
    M_real = np.zeros((100, 3), dtype="float32")
    M_real[:10, 0] = 1; M_real[:20, 1] = 1; M_real[:50, 2] = 1   # del17p smallest
    rows = []
    for cond, imp in [("real", 0.002), ("permuted", 0.001), ("random", 0.0)]:
        for s in sub:
            rows.append({"condition": cond, "subtype": s, "mean_improvement": imp})
    d = _decide(pd.DataFrame(rows), sub, M_real)
    assert d["passes"] is False
    assert "REGULARIZATION" in d["verdict"]


def test_decision_passes_when_real_beats_controls():
    sub = ["del17p", "del1p", "amp1q"]
    M_real = np.zeros((100, 3), dtype="float32")
    M_real[:10, 0] = 1; M_real[:20, 1] = 1; M_real[:50, 2] = 1
    rows = []
    for cond, imp in [("real", 0.05), ("permuted", 0.0), ("random", -0.01)]:
        for s in sub:
            rows.append({"condition": cond, "subtype": s, "mean_improvement": imp})
    d = _decide(pd.DataFrame(rows), sub, M_real)
    assert d["passes"] is True and "BIOLOGY" in d["verdict"]
