"""Part II Stage 5 — the decisive falsification gate, as a gated assertion.

The gate is pre-registered: the trained flow is PROMOTED over static only if it beats BOTH
StaticDynamics AND a random/frozen-field ablation on paired ΔC (Uno IPCW) with the CI lower
bound > 0; otherwise the pipeline reports the NULL and keeps static. This test does not
retrain (too heavy for CI) — it verifies the gate LOGIC and that the saved verdict is
internally consistent, so a future run can never silently 'promote' without the CI evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "outputs/flow/flow_gate.json"


def _promote(beats_static: bool, beats_random: bool) -> bool:
    """The pre-registered promotion rule (mirrors flow_trainer.main)."""
    return bool(beats_static and beats_random)


def test_gate_rule_requires_beating_static_and_random():
    assert _promote(True, True) is True
    assert _promote(True, False) is False
    assert _promote(False, True) is False
    assert _promote(False, False) is False


@pytest.mark.skipif(not GATE.exists(), reason="flow gate not run (heavy training; artifact gitignored)")
def test_saved_gate_verdict_is_internally_consistent():
    r = json.loads(GATE.read_text())
    # promotion must equal the AND of the two sub-gates
    assert r["PROMOTE_dynamic"] == _promote(r["gate_flow_beats_static"], r["gate_flow_beats_random_field"])
    # each sub-gate must match its CI lower bound (index 1 = 2.5th percentile) being > 0
    assert r["gate_flow_beats_static"] == (r["dC_dynamic_vs_static"][1] > 0)
    assert r["gate_flow_beats_random_field"] == (r["dC_dynamic_vs_random_field"][1] > 0)
    # if promoted, BOTH ΔC CIs must clear 0 — never promote on a null
    if r["PROMOTE_dynamic"]:
        assert r["dC_dynamic_vs_static"][1] > 0 and r["dC_dynamic_vs_random_field"][1] > 0
