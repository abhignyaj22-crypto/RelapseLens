"""Result Set A guard — the longitudinal-substrate characterization must keep reporting the real,
honest failure on open GDC data. Locks the decisive numbers so a data/plumbing change can't silently
flip the substrate verdict from FAIL to a fabricated PASS.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.figures.result_set_A import compute_result_set_a, MIN_VISITS

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real" / "clinical_survival.csv"


@pytest.fixture(scope="module")
def rep():
    return compute_result_set_a(pd.read_csv(REAL))


def test_open_tier_is_baseline_only(rep):
    v = rep["visits_per_patient"]
    assert rep["n_patients"] == 995
    assert v["median"] == 1.0
    assert v["frac_multi_visit"] == 0.0 and v["frac_ge3_visits"] == 0.0


def test_failure_condition_is_met_and_verdict_is_fail(rep):
    assert rep["failure_condition"]["met"] is True
    assert rep["failure_condition"]["median_visits_per_patient"] < MIN_VISITS
    assert rep["verdict"].startswith("FAIL")


def test_endpoint_is_os_only(rep):
    assert rep["event_prevalence"]["os_n_events"] == 191
    assert rep["event_prevalence"]["pfs_populated_frac"] == 0.0        # no serial progression endpoint


def test_treatment_line_is_baseline_not_serial(rep):
    # line_of_therapy exists but is one value/patient — must be reported, and must not be mistaken for
    # serial follow-up (the note carries that caveat).
    assert isinstance(rep["treatment_line_distribution"], dict)
    assert "not a regimen start/stop timeline" in rep["treatment_note"]
