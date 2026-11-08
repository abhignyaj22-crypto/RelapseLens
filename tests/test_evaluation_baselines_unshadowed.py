"""A1 1.5 — evaluation.baselines was a file (baselines.py) shadowed by a package (baselines/), so its
functional drug-response baselines (incl. the DeepCDR stub Phase 2.2 needs) were unreachable. 1.5 moved
the file into the package as `predictors.py` and re-exported it. These tests lock that both the class-based
baselines AND the functional predictors are now importable from the one package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import resistancemap.evaluation.baselines as B


def test_package_exposes_both_class_and_functional_baselines():
    # class-based (were always reachable)
    assert hasattr(B, "ClinicalOnlyBaseline") and hasattr(B, "SimpleSurvivalBaseline")
    # functional (were SHADOWED before 1.5 — must now be reachable)
    for fn in ("predict_global_mean", "predict_per_drug_mean", "ridge_on_esm2_meanpool",
               "predict_base_rate", "logistic_on_esm2", "deepcdr_reimpl", "require_baseline_row"):
        assert hasattr(B, fn), f"functional baseline {fn} not reachable from the package"
    assert "deepcdr" in B.BASELINES_REGRESSION and "deepcdr" in B.BASELINES_CLASSIFICATION


def test_functional_baseline_runs():
    tr = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    te = pd.DataFrame({"y": [0.0, 0.0]})
    out = B.predict_global_mean(tr, te, "y")
    assert out.shape == (2,) and np.allclose(out, 2.0)


def test_deepcdr_is_real_and_refuses_to_fabricate():
    # sota-workaround A3 filled the stub with a real DeepCDR-style MLP (structure x expression).
    # It must (a) refuse to fabricate when given no real features, (b) return real predictions otherwise.
    with pytest.raises((ValueError, NotImplementedError)):
        B.deepcdr_reimpl(None, None)                              # no inputs -> refuse, never a constant
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"feat_0": rng.normal(size=40), "feat_1": rng.normal(size=40), "ln_ic50": rng.normal(size=40)})
    te = pd.DataFrame({"feat_0": rng.normal(size=8), "feat_1": rng.normal(size=8), "ln_ic50": rng.normal(size=8)})
    out = B.deepcdr_reimpl(tr, te)
    assert out.shape == (8,) and np.all(np.isfinite(out))        # real computed predictions
