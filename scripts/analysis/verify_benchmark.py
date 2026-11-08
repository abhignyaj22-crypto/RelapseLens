"""Verify the frozen MMTTE falsification benchmark artifact (regression + honesty guard).

Reads the tracked outputs/falsification_open_gdc_os/benchmark_result.json and asserts the
pre-registered control invariants (docs/benchmark_protocol.md) still hold and that the
benchmark's own evidence does not license SOTA/superiority language. Exits nonzero on any
violation so `make benchmark-verify` / CI fail loudly if a code change silently moves the
result (e.g. reintroduces leakage, or flips the retained null).

The checks are a PURE function so they are unit-tested without re-running the pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "outputs" / "falsification_open_gdc_os" / "benchmark_result.json"
EXPECTED_SCHEMA = "1.0"
EXPECTED_ARMS = {"clinical", "clinical+omics", "residual_risk", "clinical+cyto+omics",
                 "pooled_neural", "scrambled_subtype", "permuted_omics", "random_label"}


def _uno(result, arm):
    for r in result.get("leaderboard", []):
        if r.get("arm") == arm:
            return r.get("mean_uno_cindex")
    return None


def check_benchmark_result(result: dict, expected_schema: str = EXPECTED_SCHEMA) -> dict:
    """Return {passed, checks:[(name, ok, detail)], violations:[...]} — no I/O, no pipeline."""
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    add("schema_version", result.get("schema_version") == expected_schema,
        f"{result.get('schema_version')} (expected {expected_schema})")

    arms = {r.get("arm") for r in result.get("leaderboard", [])}
    add("all_arms_present", EXPECTED_ARMS <= arms, f"missing {sorted(EXPECTED_ARMS - arms)}")

    # (i) pipeline sanity: random labels must sit at chance (else leakage regression)
    rc = _uno(result, "random_label")
    add("random_label_at_chance", rc is not None and 0.40 <= rc <= 0.60,
        f"random_label Uno-C = {rc}")

    # (ii) subtype is not biology: real subtype ~ scrambled (else the null would flip)
    real_sub, scram = _uno(result, "clinical+cyto+omics"), _uno(result, "scrambled_subtype")
    gap = None if (real_sub is None or scram is None) else abs(real_sub - scram)
    add("subtype_equals_scramble", gap is not None and gap < 0.03,
        f"|real - scrambled| = {gap}")

    # (iii) decision internal consistency: FDR-significant deltas <=> not "H0 retained"
    any_sig = any(d.get("fdr_significant") for d in result.get("paired_deltas", []))
    overall = str(result.get("decision", {}).get("overall", ""))
    h0 = "H0 retained" in overall
    # H0-retained iff no delta is FDR-significant; anything else is a stale/tampered verdict
    add("decision_consistent_with_deltas", any_sig != h0,
        f"any_fdr_significant={any_sig}, overall={overall!r}")

    # (iv) honesty gate: the benchmark's own evidence must NOT license SOTA prose (no external
    # validation is possible on open data), so a superiority claim is correctly refused.
    from mm_tte_survival.evaluation.claim_gate import benchmark_honesty_lint
    best = next((d for d in result.get("paired_deltas", [])
                 if d.get("comparison") == "clinical+omics_vs_clinical"), {})
    evidence = {"matched_cohort": True, "same_endpoint": True,
                "delta_ci_low": best.get("delta_ci_low", float("nan")),
                "fdr_significant": bool(best.get("fdr_significant")),
                "external_validation": False}
    lint = benchmark_honesty_lint("Our model achieves state-of-the-art and beats SOTA.", evidence)
    add("honesty_gate_refuses_sota", lint["allowed"] is False,
        f"licensed_for_sota={lint['licensed_for_sota_language']}")

    violations = [f"{n}: {d}" for (n, ok, d) in checks if not ok]
    return {"passed": not violations, "checks": checks, "violations": violations}


def main() -> int:
    if not ARTIFACT.exists():
        print(f"MISSING {ARTIFACT}\n  run `make benchmark` (or `mm-tte falsify`) first.", file=sys.stderr)
        return 2
    result = json.loads(ARTIFACT.read_text())
    rep = check_benchmark_result(result)
    print(f"MMTTE benchmark verify — {ARTIFACT.relative_to(ROOT)}")
    for name, ok, detail in rep["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\nOVERALL: {'PASS' if rep['passed'] else 'FAIL'}  "
          f"(decision: {result.get('decision', {}).get('overall', '?')})")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
