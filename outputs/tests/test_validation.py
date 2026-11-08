"""Unit tests for the subtype-label validation stack.

Fixtures here are tiny constructed arrays used purely to exercise the metric and
decision LOGIC — they are not research data and produce no scientific result.
"""
import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Regression: the corrected losses.py must still expose the torch primitives the
# HSS trainer needs (cox_survival_curve / survival_curve_distill). This import
# was silently broken by the losses drop-in.
# --------------------------------------------------------------------------- #
def test_trainer_and_losses_import_and_primitives_work():
    torch = pytest.importorskip("torch")
    from mm_tte_survival.training.losses import cox_survival_curve, survival_curve_distill
    import mm_tte_survival.training.trainer_hss  # must import without error
    import mm_tte_survival.experiments_regularization  # depends on trainer_hss
    eta = torch.tensor([0.0, 1.0])
    H = torch.tensor([0.1, 0.5, 1.0])
    S = cox_survival_curve(eta, H)
    assert S.shape == (2, 3)
    assert (S <= 1.0).all() and (S >= 0.0).all()
    # higher risk (eta=1) => lower survival at every grid point
    assert (S[1] <= S[0]).all()
    assert float(survival_curve_distill(S, S)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Shared surrogate caller
# --------------------------------------------------------------------------- #
def test_surrogate_caller_prevalence_and_alias():
    from mm_tte_survival.validation.surrogate_caller import call_translocation, TRANSLOC
    rng = np.random.default_rng(0)
    # CCND1 high in 20 of 100 samples
    expr = pd.DataFrame({"CCND1": rng.normal(0, 1, 100)})
    expr.loc[:19, "CCND1"] += 6.0
    calls, score, present = call_translocation(expr, ["CCND1"], 0.16)
    assert present == ["CCND1"]
    assert abs(calls.mean() - 0.16) < 0.05         # prevalence ~ target
    assert calls.loc[:19].mean() > calls.loc[20:].mean()
    # WHSC1 alias is recognised for t(4;14) even when NSD2 is absent
    expr2 = pd.DataFrame({"WHSC1": rng.normal(0, 1, 100)})
    _, _, present2 = call_translocation(expr2, TRANSLOC["t_4_14"][0], 0.13)
    assert "WHSC1" in present2


# --------------------------------------------------------------------------- #
# External GEO helpers
# --------------------------------------------------------------------------- #
def test_chrom_parser_and_log_detection():
    from mm_tte_survival.validation.external_geo import _chrom_of, to_log
    assert _chrom_of("13q14.2") == "13"
    assert _chrom_of("Xp11") == "X"
    assert _chrom_of("") == ""
    linear = pd.DataFrame(np.array([[100.0, 2000.0], [50.0, 800.0]]))
    logd = pd.DataFrame(np.array([[3.0, 8.0], [2.0, 7.0]]))
    assert to_log(linear).values.max() < 12      # log-compressed
    assert np.allclose(to_log(logd).values, logd.values)  # already log -> unchanged


def test_binary_metrics_perfect_and_random():
    from mm_tte_survival.validation.external_geo import binary_metrics
    y = np.array([0, 0, 1, 1, 1])
    perfect = binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9, 1.0]))
    assert perfect["auc"] == pytest.approx(1.0)
    assert perfect["sens"] == pytest.approx(1.0)
    degenerate = binary_metrics(np.zeros(5), np.arange(5.0))  # no positives
    assert np.isnan(degenerate["auc"])


# --------------------------------------------------------------------------- #
# FISH-ready harness
# --------------------------------------------------------------------------- #
def test_binary_agreement_and_fish_ready_inert(tmp_path):
    from mm_tte_survival.validation.fish_ready import binary_agreement, run_fish_ready
    m = binary_agreement(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert m["sens"] == pytest.approx(0.5) and m["spec"] == pytest.approx(1.0)
    # inert when no FISH file is configured
    assert run_fish_ready({"paths": {"fish": None, "outdir": str(tmp_path)}}) is None


# --------------------------------------------------------------------------- #
# Label-noise decision logic
# --------------------------------------------------------------------------- #
def test_label_noise_flip_rate_and_decision():
    from mm_tte_survival.experiments_label_noise import _flip, _decide
    rng = np.random.default_rng(1)
    col = np.zeros(10000, dtype="float32")
    flipped = _flip(col, 0.30, rng)
    assert abs(flipped.mean() - 0.30) < 0.02       # ~30% flipped to 1
    # robust: pooled never worse than independent across draws
    rows = []
    for d in range(1, 6):
        rows.append({"subtype": "del17p", "kind": "flipped", "ibs_pooled": 0.15,
                     "ibs_independent": 0.16, "pooled_minus_independent": -0.01})
    rows.append({"subtype": "del17p", "kind": "real", "ibs_pooled": 0.15,
                 "ibs_independent": 0.16, "pooled_minus_independent": -0.01})
    dec = _decide(pd.DataFrame(rows), ["del17p"])
    assert dec["robust"] is True and "ROBUST" in dec["verdict"]


# --------------------------------------------------------------------------- #
# Subtype-calibration one-shot
# --------------------------------------------------------------------------- #
def test_d_calibration_uniform_vs_skewed():
    from mm_tte_survival.experiments_calibration_subtype import d_calibration
    rng = np.random.default_rng(2)
    well = rng.uniform(0, 1, 2000); e = np.ones(2000, int)
    _, p_well = d_calibration(well, e)
    skewed = rng.uniform(0, 0.2, 2000)               # all low S(T) -> miscalibrated
    _, p_bad = d_calibration(skewed, e)
    assert p_well > 0.05 and p_bad < 0.01


def test_calibration_decision_unpromotable_even_when_internal_pass():
    from mm_tte_survival.experiments_calibration_subtype import _decide
    M = np.zeros((100, 2), dtype="float32"); M[:10, 0] = 1; M[:20, 1] = 1
    summ = pd.DataFrame([
        {"subtype": "del17p", "scheme": "pooled", "mean_ibs": 0.18},
        {"subtype": "del17p", "scheme": "real", "mean_ibs": 0.15},
        {"subtype": "del17p", "scheme": "scramble", "mean_ibs": 0.18},
        {"subtype": "del1p", "scheme": "pooled", "mean_ibs": 0.17},
        {"subtype": "del1p", "scheme": "real", "mean_ibs": 0.16},
        {"subtype": "del1p", "scheme": "scramble", "mean_ibs": 0.17},
    ])
    dec = _decide(summ, M, ["del17p", "del1p"])
    assert dec["internal_pass"] is True
    assert dec["promotable"] is False               # external replication unmeetable
    assert "CANNOT be promoted" in dec["verdict"]
