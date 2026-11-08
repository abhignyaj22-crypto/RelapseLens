"""Tests for the Phase-6 submission harness (pure verdict + predict-wrapper contract)."""
import numpy as np
import pandas as pd

from mm_tte_survival.benchmark_submit import submission_verdict, feature_predict
from mm_tte_survival.comparators import PenalizedCox


def test_verdict_no_improvement_when_ci_overlaps_zero():
    v = submission_verdict({"arm": "x"},
                           {"delta_ci_low": -0.03, "fdr_significant": False},
                           {"licensed_for_sota_language": False})
    assert not v["credited"] and not v["beats_baseline_ci"]
    assert "No evidence" in v["message"]


def test_verdict_provisional_when_ci_positive_and_fdr_but_no_external():
    v = submission_verdict({"arm": "x"},
                           {"delta_ci_low": 0.02, "fdr_significant": True},
                           {"licensed_for_sota_language": False})   # no external validation
    assert v["credited"] and v["beats_baseline_ci"] and v["fdr_significant"]
    assert not v["sota_licensed"]
    assert "PROVISIONAL" in v["message"]


def test_verdict_requires_both_ci_and_fdr():
    # FDR significant but CI overlaps 0 -> not credited
    v = submission_verdict({}, {"delta_ci_low": -0.01, "fdr_significant": True},
                           {"licensed_for_sota_language": False})
    assert not v["credited"]


def test_feature_predict_contract_returns_two_aligned_risk_vectors():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n),
                       "time_months": rng.uniform(1, 40, n),
                       "event": (rng.random(n) < 0.6).astype(int)})
    train, test = df.iloc[:40], df.iloc[40:]
    predict = feature_predict(PenalizedCox)
    r_tr, r_te = predict(train, test)
    assert len(r_tr) == len(train) and len(r_te) == len(test)
    assert np.isfinite(np.asarray(r_te, dtype=float)).all()
