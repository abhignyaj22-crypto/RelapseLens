"""sota-workaround B2 — landmark-OS dynamic(LOCF) vs static(baseline) Cox on the real serial table.

Pins the apples-to-apples DESIGN (not the ~1min real run): same at-risk cohort per landmark, dynamic = most-
recent LOCF lab, static = baseline lab, patient-disjoint folds, Uno-IPCW td-C with a paired-bootstrap CI gate.
Measured verdict: NULL — dynamic does NOT beat static on OS td-C at any landmark (t_L=12mo Δ+0.070 [−0.014,
+0.162], CI includes 0); the static clinical ceiling holds (see docs/sota_workaround/log.md). Real data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_AR = ROOT / "scripts" / "autoresearch"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))
sys.path.insert(0, str(ROOT / "src"))


def test_build_cohort_at_risk_and_locf_vs_baseline():
    import run_landmark_os as L
    from mm_tte_survival.data.serial_clinical import build_patient_visit_table
    # one at-risk patient (event well beyond t_L) with a rising albumin; one not-at-risk (early event)
    raw = pd.DataFrame([
        {"patient_id": "A", "visit_t_days": 0, "albumin_g_dl": 30.0, "event_t_days": 900, "event_type": 1},
        {"patient_id": "A", "visit_t_days": 60, "albumin_g_dl": 45.0, "event_t_days": 900, "event_type": 1},  # within t_L=91d
        {"patient_id": "B", "visit_t_days": 0, "albumin_g_dl": 33.0, "event_t_days": 30, "event_type": 1},
    ])
    v = build_patient_visit_table(raw, provenance={"source": "t", "is_real": True})
    D = L.build_cohort(v, tL=3 * L.MONTH)                      # ~91 days
    assert set(D["patient_id"]) == {"A"}                        # B (event at 30d) not at-risk beyond 91d
    row = D.iloc[0]
    assert row["dyn_albumin_g_dl"] == 45.0                      # most-recent <= t_L (LOCF)
    assert row["base_albumin_g_dl"] == 30.0                     # earliest (baseline)
    assert row["time"] == 900 - 3 * L.MONTH and row["event"] == 1


def test_design_is_paired_ci_gated_and_same_cohort():
    src = (_AR / "run_landmark_os.py").read_text()
    assert "uno_c_index" in src and "patient" in src.lower()    # Uno-IPCW td-C, patient-disjoint
    assert "ci[0] > 0" in src                                   # CAPABILITY gate = CI excludes 0
    assert "dyn_" in src and "base_" in src                     # dynamic vs static on the SAME rows
    assert "NULL" in src                                        # honest null path exists
