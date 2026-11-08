"""Part II Stage 3 — public disease-stage marginals (GSE271107).

Pure-helper tests always run. The artifact tests run only when the atlas has been built
locally (the .npy clouds are gitignored / not in CI) — they verify four non-empty stage
clouds aligned to the CoMMpass gene space with an identifiable MM endpoint.
"""
from __future__ import annotations

import numpy as np
import pytest

from mm_tte_survival.data import stage_marginals as sm


# ---- pure helpers (always run) ------------------------------------------
@pytest.mark.parametrize("fname,stage", [
    ("GSM8369863_HD1.h5", "HD"),
    ("GSM8369867_HD5.h5", "HD"),
    ("GSM8369868_MGUS_1.h5", "MGUS"),
    ("GSM8369874_SMM_1.h5", "SMM"),
    ("GSM8369878_MM_1.h5", "MM"),
    ("GSM_unknown_XYZ.h5", None),
])
def test_stage_of_maps_filenames(fname, stage):
    assert sm._stage_of(fname) == stage


def test_normalize_is_cp10k_log1p_and_zero_safe():
    counts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype="float32")
    z = sm._normalize(counts)
    assert np.isfinite(z).all()                      # zero-total row must not divide by zero
    assert (z[0] == 0).all()                         # all-zero cell stays zero
    # a value equals log1p(count/total*1e4): row1 col2 = log1p(2/4*1e4)
    assert np.isclose(z[1, 2], np.log1p(2 / 4 * 1e4), atol=1e-3)


# ---- artifact tests (skip if not built locally) -------------------------
_BUILT = (sm.OUT / "manifest.json").exists()
skip_unbuilt = pytest.mark.skipif(not _BUILT, reason="stage marginals not built (gitignored atlas)")


@skip_unbuilt
def test_four_nonempty_stage_clouds_in_commpass_space():
    import json
    man = json.loads((sm.OUT / "manifest.json").read_text())
    clouds = sm.load_stage_marginals()
    assert set(clouds) == set(sm.STAGES)
    G = len(np.load(sm.OUT / "genes.npy", allow_pickle=True))
    for s in sm.STAGES:
        assert clouds[s].shape[0] > 0, f"empty stage {s}"
        assert clouds[s].shape[1] == G == man["n_genes"]


@skip_unbuilt
def test_mm_endpoint_is_identifiable():
    import json
    man = json.loads((sm.OUT / "manifest.json").read_text())
    # honest exit criterion: MM separates from HD in plasma-cell markers (pre-malignant may overlap)
    assert man["endpoint_identifiable"] is True
    assert man["endpoint_separation_MM_over_HD"] >= 1.5
