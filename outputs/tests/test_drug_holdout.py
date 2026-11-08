"""sota-workaround A2 — leave-drug-out drug-structure fusion: the honest workaround for the 2.4 NULL.

Pins the DESIGN that makes the claim honest (not the ~1min run): folds are drug-disjoint (GroupKFold on
drug), the gate requires fusion to beat BOTH the global-mean floor AND a fingerprint-scrambled null with
CI excluding 0, and it reuses the shared R² helper. Measured verdict: CAPABILITY — fusion − floor
+0.189 [+0.104,+0.276], fusion − scrambled-fp null +1.472 [+0.735,+2.724] (see docs/sota_workaround/log.md).
Real data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_AR = ROOT / "scripts" / "autoresearch"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))


def test_is_leave_drug_out_with_scrambled_fp_null():
    src = (_AR / "run_drug_holdout.py").read_text()
    assert "GroupKFold" in src and 'groups' in src and "drug" in src   # drug-disjoint folds
    assert "permuted_fp" in src and "perm" in src                      # fingerprint-scrambled null control
    # honest gate: beat the floor AND the null, both CI-excluding-0
    assert "floor_ci[0] > 0 and null_ci[0] > 0" in src
    assert "from run_drug_gdsc2 import _r2" in src                     # reuse, no duplication


def test_floor_is_global_mean_for_unseen_drug():
    # the honest floor: an unseen drug has no per-drug mean, so global mean IS the floor (documented)
    src = (_AR / "run_drug_holdout.py").read_text()
    assert "global_floor" in src and "per-drug-mean = global mean" in src
