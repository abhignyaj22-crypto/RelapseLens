"""Tests for the Phase-5 frozen-benchmark verify guard (pure checks, no pipeline run)."""
import copy
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "verify_benchmark",
    Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "verify_benchmark.py")
verify_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_benchmark)
check = verify_benchmark.check_benchmark_result


def _good():
    arms = {"clinical": 0.726, "clinical+omics": 0.751, "residual_risk": 0.746,
            "clinical+cyto+omics": 0.746, "pooled_neural": 0.707, "scrambled_subtype": 0.742,
            "permuted_omics": 0.695, "random_label": 0.486}
    return {
        "schema_version": "1.0",
        "leaderboard": [{"arm": a, "mean_uno_cindex": v} for a, v in arms.items()],
        "paired_deltas": [{"comparison": "clinical+omics_vs_clinical",
                           "delta_ci_low": -0.021, "fdr_significant": False}],
        "decision": {"overall": "H0 retained: no biological structure survives the controls"},
    }


def _names_failed(rep):
    return {n for (n, ok, _d) in rep["checks"] if not ok}


def test_frozen_like_artifact_passes():
    rep = check(_good())
    assert rep["passed"], rep["violations"]


def test_random_label_leakage_is_caught():
    r = _good()
    for row in r["leaderboard"]:
        if row["arm"] == "random_label":
            row["mean_uno_cindex"] = 0.90        # leakage: random labels should be ~0.5
    rep = check(r)
    assert not rep["passed"] and "random_label_at_chance" in _names_failed(rep)


def test_missing_arm_is_caught():
    r = _good()
    r["leaderboard"] = [x for x in r["leaderboard"] if x["arm"] != "permuted_omics"]
    rep = check(r)
    assert "all_arms_present" in _names_failed(rep)


def test_subtype_biology_flip_is_caught():
    r = _good()
    for row in r["leaderboard"]:
        if row["arm"] == "scrambled_subtype":
            row["mean_uno_cindex"] = 0.60        # real >> scrambled would mean subtype biology
    rep = check(r)
    assert "subtype_equals_scramble" in _names_failed(rep)


def test_stale_decision_vs_significant_delta_is_caught():
    r = _good()
    r["paired_deltas"][0]["fdr_significant"] = True   # significant, but verdict still says H0
    rep = check(r)
    assert "decision_consistent_with_deltas" in _names_failed(rep)


def test_wrong_schema_is_caught():
    r = _good()
    r["schema_version"] = "0.9"
    assert "schema_version" in _names_failed(check(r))


def test_honesty_gate_refuses_sota_on_open_data():
    # even if the increment CI were positive, no external validation -> SOTA prose refused
    r = _good()
    r["paired_deltas"][0]["delta_ci_low"] = 0.05
    r["paired_deltas"][0]["fdr_significant"] = True
    r["decision"]["overall"] = "H1 (some structure survives)"   # keep decision consistent
    rep = check(r)
    assert "honesty_gate_refuses_sota" not in _names_failed(rep)
