"""sota-workaround D1 — two-stage joint model (MixedLM BLUP shrinkage) design contract + BLUP mechanics.

Unit-tests the closed-form BLUP (shrinks a noisy single-patient trajectory toward the population; more data ->
less shrinkage) and pins the leak-free, triple-gated, whole-sweep-BH design. Measured verdict in
docs/sota_workaround/log.md. Real data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_AR = ROOT / "scripts" / "autoresearch"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))
sys.path.insert(0, str(ROOT / "src"))


def test_blup_shrinks_toward_population():
    import run_joint_model as J
    beta = np.array([0.0, 0.0])                     # population: flat at 0
    D = np.array([[1.0, 0.0], [0.0, 1.0]])          # unit random-effect variance
    # a patient with ONE noisy high point -> BLUP intercept shrunk well below the raw value
    eff1, var1 = J._blup_feats(np.array([0.0]), np.array([10.0]), beta, D, sig2=1.0)
    assert 0.0 < eff1[0] < 10.0                     # shrunk toward population, not the raw 10
    # more consistent observations -> less shrinkage (BLUP moves closer to the data mean)
    eff_many, _ = J._blup_feats(np.array([0.0, 0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0, 10.0]), beta, D, 1.0)
    assert eff_many[0] > eff1[0]                     # strength borrowed from repeated evidence
    # zero observations -> exactly the population effect (no fabrication)
    eff0, var0 = J._blup_feats(np.array([]), np.array([]), beta, D, 1.0)
    assert np.allclose(eff0, beta) and var0 == 0.0


def test_design_is_leakfree_bh_wholesweep_gated():
    src = (_AR / "run_joint_model.py").read_text()
    assert "MixedLM" in src or "mixedlm" in src                  # random-intercept+slope trajectory sub-model
    assert "re_formula=\"~t\"" in src                            # random slope
    assert 'long["fold"] != f' in src                           # LMM fit on TRAIN patients only (leak-free)
    assert "PRIOR_SWEEP_P" in src and "whole" in src.lower()     # BH across the ENTIRE dynamic sweep
    assert 'E["ci_near"][0] > 0' in src and "p_swp < 0.05" in src and "scrambled" in src.lower()
    assert "NULL" in src and "CAPABILITY" in src
