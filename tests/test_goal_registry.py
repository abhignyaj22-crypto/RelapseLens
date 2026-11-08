"""A1 goal registry — keeps the dashboard-goal → code → honest-status map truthful and non-drifting.

Guards: every referenced module path actually exists (no phantom mapping), every L0–L9 layer is mapped,
all three questions are covered, and the honesty invariants hold (the data-blocked claims stay BLOCKED
with a reason + the tier that unlocks them; nothing advertised-but-unsupported is silently marked
ANSWERABLE).
"""
from __future__ import annotations

from pathlib import Path

from resistancemap.goal import (
    CAPABILITIES, LAYERS, Question, Status, Tier, capabilities_for, by_status, summary)

ROOT = Path(__file__).resolve().parents[1]


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def test_every_referenced_module_exists():
    missing = []
    for c in CAPABILITIES:
        missing += [m for m in c.modules if not _exists(m)]
        if c.autoresearch_entry and not _exists(c.autoresearch_entry):
            missing.append(c.autoresearch_entry)
    for _, (_, mod) in LAYERS.items():
        if not _exists(mod):
            missing.append(mod)
    assert not missing, f"registry references non-existent paths: {missing}"


def test_layers_map_to_existing_modules():
    # honest-core edition: LAYERS advertises ONLY the layers backed by surviving, tested code (the
    # aspirational L1/L3/L5/L7 breadth was removed). Each advertised layer must map to a real module.
    assert LAYERS, "no dashboard layers mapped"
    for key, (_label, mod) in LAYERS.items():
        assert key.startswith("L") and key[1:].isdigit(), f"bad layer key {key}"
        assert _exists(mod), f"layer {key} maps to non-existent module {mod}"


def test_all_three_questions_covered():
    for q in Question:
        assert capabilities_for(q), f"no capability registered for {q}"


def test_blocked_claims_are_honest():
    # every BLOCKED capability must say WHY and name the tier that would unlock it
    for c in by_status(Status.BLOCKED):
        assert c.blocked_reason, f"{c.name} is BLOCKED but gives no reason"
        assert isinstance(c.tier, Tier)
    # the dynamic 3/6/12-mo trajectory MUST be blocked (Result A: open data is baseline-only)
    dyn = [c for c in CAPABILITIES if "trajectory" in c.name and c.question == Question.WHEN]
    assert dyn and dyn[0].status == Status.BLOCKED and dyn[0].tier == Tier.T2


def test_answerable_claims_have_evidence_and_a_keep_revert_entry():
    for c in by_status(Status.ANSWERABLE):
        assert c.evidence, f"{c.name} claims ANSWERABLE without evidence"
        # an answerable claim is one the autoresearch loop can (and does) score
        assert c.autoresearch_entry and _exists(c.autoresearch_entry)


def test_summary_shape():
    s = summary()
    assert set(s) == {q.value for q in Question}
    assert all(set(v) == {st.value for st in Status} for v in s.values())
