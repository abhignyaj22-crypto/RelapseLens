"""Tests for the Hierarchical Subtype Survival (HSS) building blocks."""
import numpy as np
import torch
import pytest

from mm_tte_survival.models.hierarchical import HierarchicalSubtypeSurvival
from mm_tte_survival.training.losses import cox_survival_curve, survival_curve_distill, soft_distillation_loss
from mm_tte_survival.survival_curves import breslow_baseline, cox_survival, time_grid, ipcw_ibs
from mm_tte_survival.data.splits import (patient_hash_split, assert_patient_disjoint,
                                         assert_one_row_per_patient)


def test_hss_forward_shapes_with_and_without_subtypes():
    x = torch.randn(20, 7)
    for n_sub in (0, 3):
        m = (torch.rand(20, n_sub) > 0.5).float()
        out = HierarchicalSubtypeSurvival(7, n_sub, head_type="cox")(x, m)
        assert out["mixed"].shape == (20, 1)
        assert out["weights"].shape == (20, n_sub + 1)
        # agnostic always present -> weights sum to 1 and first column > 0
        assert torch.allclose(out["weights"].sum(1), torch.ones(20), atol=1e-5)


def test_pure_agnostic_fallback_when_no_membership():
    x = torch.randn(8, 5)
    m = torch.zeros(8, 3)                       # no abnormality called
    model = HierarchicalSubtypeSurvival(5, 3, head_type="cox")
    out = model(x, m)
    # all weight on the agnostic head -> mixed == agnostic
    assert torch.allclose(out["mixed"], out["agnostic"], atol=1e-5)


def test_soft_distillation_temperature_removed():
    # the no-op temperature kwarg must be gone (call must work positionally only)
    s, t = torch.randn(10), torch.randn(10)
    val = soft_distillation_loss(s, t)
    assert torch.isfinite(val)
    with pytest.raises(TypeError):
        soft_distillation_loss(s, t, 2.0)


def test_curve_distill_zero_when_identical():
    eta = torch.randn(12)
    H = torch.linspace(0.1, 1.0, 6)
    S = cox_survival_curve(eta, H)
    assert S.shape == (12, 6)
    assert survival_curve_distill(S, S).item() == pytest.approx(0.0, abs=1e-7)


def test_breslow_and_ipcw_ibs_run():
    rng = np.random.default_rng(0)
    t = rng.uniform(1, 40, 60); e = (rng.random(60) > 0.5).astype(int); eta = rng.normal(size=60)
    grid = time_grid(t, e, t)
    assert grid is not None
    H = breslow_baseline(t, e, eta, grid)
    assert H.shape == grid.shape and np.all(np.diff(H) >= -1e-9)   # non-decreasing
    ibs = ipcw_ibs(t, e, t, e, cox_survival(eta, H), grid)
    assert 0.0 <= ibs <= 1.0


def test_patient_disjoint_helpers():
    ids = [f"p{i}" for i in range(50)]
    split = patient_hash_split(ids, seed=42)
    assert_one_row_per_patient(ids)
    assert_patient_disjoint(ids, split)            # must not raise
    with pytest.raises(AssertionError):
        assert_one_row_per_patient(ids + ["p0"])   # duplicate patient
    with pytest.raises(AssertionError):
        bad = np.array(["train", "test"]); assert_patient_disjoint(["p0", "p0"], bad)


def test_official_splitter_is_patient_disjoint_and_stratified():
    import pandas as pd
    from mm_tte_survival.data.splits import patient_disjoint_stratified_split
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(200)],
                       "event": (rng.random(200) > 0.6).astype(int)})
    split = patient_disjoint_stratified_split(df, "patient_id", "event", 42,
                                              mode="train_val_test", test_frac=0.2, val_frac=0.15)
    # patient-disjoint (one row per patient here) and event rate preserved within tol
    base = df.event.mean()
    for part in ("train", "val", "test"):
        m = (split == part).values
        assert abs(df.event[m].mean() - base) < 0.12


def test_leakage_audit_fires_on_real_violations():
    import pandas as pd
    from mm_tte_survival.evaluation.evaluate import _audit_leakage
    df = pd.DataFrame({"patient_id": ["a", "b", "c", "d"], "time_months": [1, 2, 3, 4],
                       "event": [1, 0, 1, 0], "age": [60, 70, 65, 55]})
    tm = pd.Series([True, True, False, False])
    groups = {"clinical": ["age"], "cyto": [], "omics": [], "programs": []}
    # clean (no omics): must not raise
    _audit_leakage(df, tm, {}, groups, "time_months", "event", [], False)
    # endpoint column smuggled into features -> must raise
    with pytest.raises(AssertionError):
        _audit_leakage(df, tm, {}, {**groups, "clinical": ["age", "event"]},
                       "time_months", "event", [], False)
    # duplicate patient -> must raise
    dfd = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    tmd = pd.Series([True, True, False, False, True])
    with pytest.raises(AssertionError):
        _audit_leakage(dfd, tmd, {}, groups, "time_months", "event", [], False)
    # full-cohort precomputed PC column in features -> must raise (in-fold PCA required)
    dfp = df.assign(PC1=[0.1, 0.2, 0.3, 0.4])
    with pytest.raises(AssertionError):
        _audit_leakage(dfp, tm, {}, {**groups, "omics": ["PC1"]},
                       "time_months", "event", [], True)
    # in-fold omics with a manifest that (dishonestly) reports test rows in the fit -> must raise
    dfi = df.assign(inPC1=[0.1, 0.2, 0.3, 0.4])
    bad_manifest = [{"pca_fit_on": "train_only", "test_rows_in_fit": 3, "train_test_patient_overlap": 0}]
    with pytest.raises(AssertionError):
        _audit_leakage(dfi, tm, {}, {**groups, "omics": ["inPC1"]},
                       "time_months", "event", bad_manifest, True)
