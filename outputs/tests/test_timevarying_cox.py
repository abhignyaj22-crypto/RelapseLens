"""sota-workaround C2 — time-varying (counting-process) Cox harness: interval construction + gate contract.

Unit-tests the leak-proof start-stop interval builder (LOCF covariate per interval, event only on the final
interval) and pins the triple gate. Measured verdict in docs/sota_workaround/log.md. Real data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_AR = ROOT / "scripts" / "autoresearch"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))
sys.path.insert(0, str(ROOT / "src"))


def test_intervals_are_start_stop_locf_event_on_final():
    import run_timevarying_cox as C
    from mm_tte_survival.data.serial_clinical import build_patient_visit_table
    raw = pd.DataFrame([
        {"patient_id": "A", "visit_t_days": 0, "albumin_g_dl": 30.0, "event_t_days": 300, "event_type": 1},
        {"patient_id": "A", "visit_t_days": 100, "albumin_g_dl": 35.0, "event_t_days": 300, "event_type": 1},
        {"patient_id": "A", "visit_t_days": 200, "albumin_g_dl": 40.0, "event_t_days": 300, "event_type": 1},
    ])
    v = build_patient_visit_table(raw, provenance={"source": "t", "is_real": True})
    iv = C.intervals_table(v).sort_values("start").reset_index(drop=True)
    assert list(iv["start"]) == [0.0, 100.0, 200.0] and list(iv["stop"]) == [100.0, 200.0, 300.0]
    assert list(iv["event"]) == [0, 0, 1]                       # event only on the final interval (stop==event_t)
    assert list(iv["albumin_g_dl"]) == [30.0, 35.0, 40.0]       # LOCF covariate at each interval's start visit


def test_triple_gate_and_two_model_design():
    src = (_AR / "run_timevarying_cox.py").read_text()
    assert "CoxTimeVaryingFitter" in src and "CoxPHFitter" in src            # dynamic vs static model
    assert "start_col=\"start\", stop_col=\"stop\"" in src                    # counting-process fit
    assert 'multiplicity_adjust(pvals, method="bh")' in src
    assert 'E["ci"][0] > 0' in src and "p_adj < 0.05" in src and "scramble" in src.lower()
    assert "NULL" in src and "CAPABILITY" in src
