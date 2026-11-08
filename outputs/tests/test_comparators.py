"""Tests for the comparator zoo, honesty linter, and devil's-advocate recompute."""
import numpy as np
import pandas as pd

from mm_tte_survival.comparators import (
    build_comparators, backend_report, devils_advocate_recompute,
    PenalizedCox, RandomSurvivalForestC, ConstantBaseline, _HAS_SKSURV)
from mm_tte_survival.metrics_ipcw import uno_c_index
from mm_tte_survival.evaluation.claim_gate import benchmark_honesty_lint


def _signal_cohort(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x_sig = rng.normal(size=n)          # prognostic
    x_noise = rng.normal(size=n)        # not
    risk = 0.9 * x_sig
    t = np.exp(-risk) * rng.weibull(1.3, n) * 20 + 1
    e = (rng.random(n) < 0.6).astype(int)
    df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)], "time_months": t, "event": e,
                       "x_sig": x_sig, "x_noise": x_noise})
    return df.iloc[:200].reset_index(drop=True), df.iloc[200:].reset_index(drop=True)


def test_backend_report_matches_environment():
    rep = backend_report()
    assert rep["scikit_survival"] == _HAS_SKSURV
    assert isinstance(rep["skipped"], list)          # honest manifest of absent backends


def test_comparators_fit_and_rank_correctly():
    tr, te = _signal_cohort()
    t_tr, e_tr = tr.time_months.values, tr.event.values
    t_te, e_te = te.time_months.values, te.event.values
    for comp in [PenalizedCox(["x_sig"]), ConstantBaseline(["x_sig"])] + (
            [RandomSurvivalForestC(["x_sig"])] if _HAS_SKSURV else []):
        comp.fit(tr)
        r = comp.risk(te)
        assert len(r) == len(te)
        c = uno_c_index(t_tr, e_tr, t_te, e_te, r)
        if comp.is_baseline:
            assert np.isnan(c) or abs(c - 0.5) < 0.12   # constant -> no discrimination
        else:
            assert c > 0.6                               # picks up the real signal


def test_build_comparators_includes_installed_backends():
    comps = build_comparators(["x_sig"])
    names = {c.name for c in comps}
    assert "PenalizedCox(lifelines)" in names and "OPSD-Cox(torch)" in names
    if _HAS_SKSURV:
        assert "RandomSurvivalForest(sksurv)" in names


def test_honesty_linter_gates_sota_language():
    weak = {"matched_cohort": True, "same_endpoint": True, "delta_ci_low": -0.01,
            "fdr_significant": False, "external_validation": False}
    strong = {"matched_cohort": True, "same_endpoint": True, "delta_ci_low": 0.02,
              "fdr_significant": True, "external_validation": True}
    assert benchmark_honesty_lint("our model is neutral here", weak)["allowed"]         # no SOTA term
    assert not benchmark_honesty_lint("our model outperforms the SOTA", weak)["allowed"]
    assert benchmark_honesty_lint("our model outperforms the SOTA", strong)["allowed"]


def test_devils_advocate_agrees_on_real_signal():
    tr, te = _signal_cohort()
    t_tr, e_tr = tr.time_months.values, tr.event.values
    t_te, e_te = te.time_months.values, te.event.values
    base = PenalizedCox(["x_noise"]).fit(tr)
    rich = PenalizedCox(["x_noise", "x_sig"]).fit(tr)
    headline = (uno_c_index(t_tr, e_tr, t_te, e_te, rich.risk(te))
                - uno_c_index(t_tr, e_tr, t_te, e_te, base.risk(te)))
    da = devils_advocate_recompute(tr, te, ["x_noise"], ["x_noise", "x_sig"],
                                   "time_months", "event", headline_delta=headline)
    assert da["verdict"] == "concedes" and da["sign_stable"]
