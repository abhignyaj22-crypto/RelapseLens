"""sota-workaround B1 — un-collapse GDC follow_ups into a real per-visit serial table.

Unit-tests the pure row-builder (no network): OS endpoint attached, serial labs mapped to the module schema,
and temporal hygiene (drop visits after the event and pre-diagnosis negative-time visits) so the table passes
validate_temporal_order. Measured on the real fetch: 994 patients, 8272 visits, median 9 visits/pt (see
docs/sota_workaround/log.md). Real data only — no synthetic visits.
"""
from __future__ import annotations

import sys
from pathlib import Path

_RD = Path(__file__).resolve().parents[1] / "scripts" / "realdata"
if str(_RD) not in sys.path:
    sys.path.insert(0, str(_RD))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_serial_from_followups as B  # noqa: E402


def _case():
    return {"submitter_id": "P1",
            "demographic": {"vital_status": "dead", "days_to_death": 200},
            "follow_ups": [
                {"days_to_follow_up": 0, "molecular_tests": [{"laboratory_test": "Albumin", "test_value": 40}]},
                {"days_to_follow_up": 100, "molecular_tests": [{"laboratory_test": "Beta 2 Microglobulin", "test_value": 3.0}]},
                {"days_to_follow_up": 300, "molecular_tests": []},   # after death (200) -> dropped
                {"days_to_follow_up": -30, "molecular_tests": []},   # pre-diagnosis negative time -> dropped
            ]}


def test_build_rows_endpoint_and_temporal_filtering():
    df = B.build_rows([_case()])
    assert set(df["visit_t_days"]) == {0.0, 100.0}          # 300 (post-event) and -30 (pre-dx) dropped
    assert (df["event_t_days"] == 200.0).all()             # OS endpoint attached
    assert (df["event_type"] == 1).all()                   # death event
    assert df.loc[df.visit_t_days == 0, "albumin_g_dl"].iloc[0] == 40.0   # GDC lab -> module schema
    assert df.loc[df.visit_t_days == 100, "b2m_mg_l"].iloc[0] == 3.0


def test_passes_temporal_order_and_is_real_provenance():
    from mm_tte_survival.data.serial_clinical import build_patient_visit_table
    df = B.build_rows([_case()])
    v = build_patient_visit_table(df, provenance={"source": "test", "is_real": True})  # would raise if order bad
    assert v.attrs["provenance"]["is_real"] is True and len(v) == 2
