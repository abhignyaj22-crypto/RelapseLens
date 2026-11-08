"""A1 2.1 — the defective-FHT head is registered as an autoresearch candidate CORE on the frozen OS scorer.

Pins the WIRING (registration + honest baseline + callable contract), not the numbers: the full run is a
36s real-data model fit judged by the pre-registered keep/revert rule (recorded in the autoresearch ledger,
which git-ignores outputs/). The measured verdict was REVERT — Uno-C 0.650 < baseline Cox 0.751 and below the
permuted-omics control 0.695 — i.e. the FHT mechanism repairs calibration, not OS rank (see docs/autoresearch/
SCORECARD.md arm 7 + Result Set C). This test guards that the arm exists and is judged against the right bar.
"""
from __future__ import annotations

import sys
from pathlib import Path

_AR = Path(__file__).resolve().parents[1] / "scripts" / "autoresearch"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))


def test_defective_fht_registered_against_honest_baseline():
    from candidates import REGISTRY  # noqa: E402
    assert "defective_fht_os" in REGISTRY
    spec = REGISTRY["defective_fht_os"]
    # judged against the SAME clinical+omics penalised-Cox bar as the other cores (no easier baseline)
    assert spec["baseline"] == "clinical+omics"
    assert callable(spec["predict"])


def test_defective_fht_predict_signature():
    import inspect
    from candidates import defective_fht_os  # noqa: E402
    params = list(inspect.signature(defective_fht_os).parameters)
    assert params[:2] == ["train_df", "test_df"]  # predict(train_df, test_df) -> (risk_tr, risk_te)
