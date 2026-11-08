"""sota-workaround B0 — GDC follow-up density probe (feasibility gate for the dynamic-when workaround).

Unit-tests the pure visit-counting logic (no network) and pins the honest gate. Measured on the REAL GDC
fetch: N=995, 765 patients with >=3 lab-carrying visits -> GO (Angle B feasible); the BLOCKED status was a
collapse artifact in fetch_clinical (first_lab), not an absence of serial data. Real data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

_RD = Path(__file__).resolve().parents[1] / "scripts" / "realdata"
if str(_RD) not in sys.path:
    sys.path.insert(0, str(_RD))
import probe_followup_density as P  # noqa: E402


def test_visit_counts_distinguishes_dated_vs_lab_carrying():
    case = {"follow_ups": [
        {"days_to_follow_up": 0, "molecular_tests": [{"laboratory_test": "Albumin", "test_value": 4.0}]},
        {"days_to_follow_up": 90, "molecular_tests": [{"laboratory_test": "Beta 2 Microglobulin", "test_value": 3.0}]},
        {"days_to_follow_up": 180, "molecular_tests": []},          # dated but no lab
        {"days_to_follow_up": None, "molecular_tests": [{"laboratory_test": "Albumin", "test_value": 4.0}]},  # undated
    ]}
    dated, lab_dated = P.visit_counts(case)
    assert dated == 3            # {0, 90, 180}
    assert lab_dated == 2        # {0, 90}


def test_no_followups_is_zero():
    assert P.visit_counts({}) == (0, 0)
    assert P.visit_counts({"follow_ups": []}) == (0, 0)


def test_gate_is_honest_and_never_fabricates():
    src = (_RD / "probe_followup_density.py").read_text()
    assert ">=100 patients with >=3 lab-carrying" in src           # explicit gate rule
    assert "BLOCKED" in src and "fabricate" in src                 # honest no-go path, no synthetic serial data
    assert "from fetch_clinical import fetch_all" in src           # uses the REAL GDC fetch
