"""sota-workaround C1 — serial-lab trajectory features (the change a static Cox cannot see).

Unit-tests the pure per-lab trend summariser (slope/delta/variability/LOCF/baseline) and the arm-column sets.
Real data only; the feature builder never fabricates a trend where there is no evidence (single/absent obs -> 0
change). The full landmark-trajectory run + its CI/BH/permutation gate live in run_landmark_trajectory.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm_tte_survival.data import trajectory_features as T  # noqa: E402


def test_lab_trajectory_slope_delta_var():
    hist = pd.DataFrame({"visit_t_days": [0, 100, 200], "albumin_g_dl": [30.0, np.nan, 45.0]})
    f = T.lab_trajectory(hist, "albumin_g_dl")
    assert f["base_albumin_g_dl"] == 30.0 and f["last_albumin_g_dl"] == 45.0
    assert f["delta_albumin_g_dl"] == 15.0                      # last - base
    assert abs(f["slope_albumin_g_dl"] - 15.0 / 200.0) < 1e-9   # OLS slope per day over the 2 real obs
    assert f["var_albumin_g_dl"] == 7.5 and f["nobs_albumin_g_dl"] == 2.0


def test_single_or_absent_obs_has_no_fabricated_trend():
    one = T.lab_trajectory(pd.DataFrame({"visit_t_days": [0], "b2m_mg_l": [3.0]}), "b2m_mg_l")
    assert one["delta_b2m_mg_l"] == 0.0 and one["slope_b2m_mg_l"] == 0.0 and one["nobs_b2m_mg_l"] == 1.0
    absent = T.lab_trajectory(pd.DataFrame({"visit_t_days": [0]}), "ldh_u_l")   # column missing
    assert np.isnan(absent["base_ldh_u_l"]) and absent["slope_ldh_u_l"] == 0.0 and absent["nobs_ldh_u_l"] == 0.0


def test_arm_columns_isolate_change():
    labs = ["albumin_g_dl", "ldh_u_l"]
    cols = T.arm_columns(labs)
    assert cols["static"] == ["base_albumin_g_dl", "base_ldh_u_l"]
    assert cols["locf"] == ["last_albumin_g_dl", "last_ldh_u_l"]
    # trajectory = baseline PLUS change features; strictly a superset of the change-only block
    assert set(T.change_columns(labs)) < set(cols["trajectory"])
    assert all(c in cols["trajectory"] for c in cols["static"])
