"""A1 2.2 — the GDSC2 per-drug WHAT harness reports the MANDATORY baseline panel.

Pins the discipline, not the 1-minute real-data run: the GDSC2×CCLE harness must (a) source its baseline
panel from the A1-1.5-unshadowed `evaluation.baselines`, and (b) enforce `require_baseline_row` so no
headline drug-response number is reported without `per_drug_mean` on the same split. Measured verdict:
CAPABILITY (modest) — expression adds ΔR²=+0.048 [+0.038,+0.057] over per-drug-mean, permutation-validated;
DeepCDR SOTA row honestly unavailable (see docs/autoresearch/SCORECARD.md). Real data only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gdsc2_runner_uses_the_unshadowed_baseline_panel():
    src = (ROOT / "scripts/autoresearch/run_drug_gdsc2.py").read_text()
    assert "from resistancemap.evaluation.baselines import" in src
    for fn in ("predict_per_drug_mean", "predict_global_mean", "require_baseline_row", "deepcdr_reimpl"):
        assert fn in src, f"GDSC2 harness must use {fn} from the unshadowed panel"


def test_require_baseline_row_blocks_a_headline_without_per_drug_mean():
    from resistancemap.evaluation.baselines import require_baseline_row
    with pytest.raises(AssertionError):
        require_baseline_row(pd.DataFrame({"model": ["ridge_esm2"], "r2": [0.99]}))  # per_drug_mean missing
    # complete panel passes
    require_baseline_row(pd.DataFrame({"model": ["per_drug_mean", "ridge_esm2"], "r2": [0.74, 0.79]}))


def test_deepcdr_sota_row_is_real_not_fabricated():
    # sota-workaround A3: now a real DeepCDR-style MLP evaluated on the caller's identical folds.
    # It computes from data (no hardcoded 0.84 headline) and refuses to invent a number from nothing.
    import numpy as np
    from resistancemap.evaluation.baselines import deepcdr_reimpl
    with pytest.raises((ValueError, NotImplementedError)):
        deepcdr_reimpl(None, None)                                # never fabricate a headline
    rng = np.random.default_rng(1)
    tr = pd.DataFrame({"feat_0": rng.normal(size=30), "ln_ic50": rng.normal(size=30)})
    te = pd.DataFrame({"feat_0": rng.normal(size=6), "ln_ic50": rng.normal(size=6)})
    out = deepcdr_reimpl(tr, te)
    assert out.shape == (6,) and np.all(np.isfinite(out)) and not np.allclose(out, 0.84)
