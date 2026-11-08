"""A1 3.1 — the honest serving surface: every response self-declares its claim-tier; blocked claims refuse.

Exercised with FastAPI TestClient (no live server, no trained weights). The decisive honesty tests: the
dynamic 3/6/12-mo trajectory is REFUSED with a 422 + substrate-gate reason (never a fabricated trajectory),
and every capability response carries its status/tier/evidence straight from goal.py.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from resistancemap.inference.honest_api import create_honest_app  # noqa: E402

client = TestClient(create_honest_app())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_capabilities_carry_status_and_tier():
    r = client.get("/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body and "when" in body["questions"]
    when = body["questions"]["when"]
    caps = when["capabilities"]
    assert caps and all({"status", "tier", "evidence"} <= set(c) for c in caps)


def test_when_declares_a_blocked_trajectory():
    env = client.get("/capabilities/when").json()
    assert env["blocked"], "WHEN must declare the blocked dynamic trajectory"


def test_unknown_question_404():
    assert client.get("/capabilities/nonsense").status_code == 404


def test_trajectory_is_refused_not_fabricated():
    r = client.post("/predict/trajectory")
    assert r.status_code == 422                       # honest refusal, not a fabricated 3/6/12mo curve
    detail = r.json()["detail"]
    assert detail["status"] == "blocked_pending_data" and detail["blocked_reason"]


def test_predict_what_and_why_serve_claim_envelopes():
    for q in ("what", "why"):
        r = client.post(f"/predict/{q}")
        assert r.status_code == 200
        assert r.json()["servable_now"]              # these questions have answerable/partial capabilities


def test_targets_are_associational():
    r = client.get("/targets/TP53")
    assert r.status_code == 200 and r.json()["claim_tier"] == "ASSOCIATIONAL"


# ---- 3.2 L0 data-validation + claim-gate ----
def test_l0_abstains_on_ood_input():
    r = client.post("/predict/when", json={"features": {"x": 1e9}})   # |x|>1e6 -> out of distribution
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["abstained"] is True and detail["reasons"]          # abstains, does not fabricate


def test_l0_passes_valid_payload_and_stamps_gate():
    r = client.post("/predict/when", json={"features": {"age": 0.4, "iss": 1.0}})
    assert r.status_code == 200
    gate = r.json()["gate"]
    assert gate is not None and gate["l0_validation"] == "passed" and gate["n_features"] == 2


def test_empty_body_is_a_capability_query_not_an_abstention():
    r = client.post("/predict/when", json={"features": {}})
    assert r.status_code == 200 and r.json()["gate"] is None          # no payload -> envelope, no gate stamp
