"""Result Set C — fast guard on the event-head mechanics (no full pipeline run).

Locks the properties that make the H2 story valid: the defective head learns a non-crossing mass, the
controlled ablation clamps it off, and every head emits a monotone survival curve in [0,1]. The full
5-fold real-data ablation lives in scripts/paper/result_set_C_fht_ablation.py.
"""
from __future__ import annotations

import numpy as np
import torch

from scripts.paper.result_set_C_fht_ablation import _Head, _fit

D, N = 6, 40


def _toy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, D)).astype("float32")
    t = (rng.random(N) * 40 + 1).astype("float32")
    e = (rng.random(N) < 0.4).astype("float32")
    return X, t, e


def test_defective_learns_noncrossing_mass_and_ablation_clamps_it():
    X, t, e = _toy()
    dfe = _fit("defective_fht", X, t, e, 0)
    no = _fit("defective_noPinf", X, t, e, 0)
    p_def = dfe.p_never(X)
    p_no = no.p_never(X)
    assert (p_def >= 0).all() and (p_def <= 1).all()
    # the ablation forces drift>0 => essentially zero non-crossing mass
    assert p_no.mean() < 1e-2
    # the free-drift head is allowed a real cure fraction (some patients get mass > 0)
    assert p_def.max() > p_no.max()


def test_all_heads_emit_monotone_survival_in_unit_interval():
    X, t, e = _toy()
    times = np.array([1.0, 5.0, 15.0, 40.0])
    for kind in ["aft", "proper_fht", "defective_fht", "defective_noPinf"]:
        m = _fit(kind, X, t, e, 0)
        S = m.survival(torch.tensor(X), times).numpy()
        assert S.shape == (N, len(times))
        assert (S >= 0).all() and (S <= 1.0 + 1e-6).all()
        # non-increasing across increasing horizons
        assert (np.diff(S, axis=1) <= 1e-5).all()


def test_defective_noPinf_is_a_strict_subcase_not_identical():
    X, t, e = _toy()
    dfe = _fit("defective_fht", X, t, e, 0)
    no = _fit("defective_noPinf", X, t, e, 0)
    # different fitted survival (the cure mass changes long-horizon survival)
    S1 = dfe.survival(torch.tensor(X), np.array([60.0])).numpy()
    S2 = no.survival(torch.tensor(X), np.array([60.0])).numpy()
    assert not np.allclose(S1, S2)
