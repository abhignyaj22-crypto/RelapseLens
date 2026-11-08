"""Honest serving surface (A1 Phase 3.1) — every response self-declares its claim-tier from goal.py.

The legacy `api.py` is the *aspirational* product surface: it needs a trained pipeline that does not exist
on disk and would serve exactly the claims Phase 2 proved BLOCKED (the 3/6/12-mo trajectory) or REFUTED (the
multimodal fusion lift). Serving it would fabricate. This surface serves what the data actually supports and
nothing else:

  GET  /health
  GET  /capabilities                 — the honest scoreboard (answerable / partial / blocked per question)
  GET  /capabilities/{question}      — capabilities + tiers for what|when|why
  POST /predict/{question}           — a ClaimEnvelope for what|when|why: status, tier, evidence, and the
                                       list of servable-now vs blocked capabilities
  POST /predict/trajectory           — REFUSED 422 with the substrate-gate reason (dynamic 3/6/12mo is
                                       BLOCKED on baseline-only open data — never a fabricated trajectory)
  GET  /targets/{gene}               — WHY graph attribution envelope (ASSOCIATIONAL)

The single source of truth is `resistancemap.goal` (the registry from A1 1.1). Blocked capabilities return
their honest `blocked_reason` with a 422; the refuted fusion lift is not exposed. No trained weights needed —
this is the governance/claim-tier layer, fully testable with FastAPI TestClient.
"""
from __future__ import annotations

import math
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from resistancemap.goal import Question, Status, capabilities_for, summary

_DISCLAIMER = ("Every claim self-declares its data-gate tier from goal.py; blocked claims are refused, "
               "not fabricated. Static clinical Cox is the honest OS-rank ceiling; the multimodal fusion "
               "lift and the dynamic 3/6/12-mo trajectory are refuted/blocked and are NOT served.")
_QMAP = {"what": Question.WHAT, "when": Question.WHEN, "why": Question.WHY}
_SHORT = {v: k for k, v in _QMAP.items()}   # Question.WHEN -> "when" (public form, not the enum value)


class CapabilityView(BaseModel):
    name: str
    layer: str
    status: str
    tier: str
    evidence: str
    blocked_reason: str | None = None
    servable_now: bool


class ClaimEnvelope(BaseModel):
    question: str
    servable_now: list[str] = Field(description="capabilities answerable/partial on the current data tier")
    blocked: list[str] = Field(description="capabilities blocked pending a higher data tier")
    capabilities: list[CapabilityView]
    gate: dict | None = Field(None, description="L0 data-validation + claim-gate stamp (present when a "
                                                "feature payload was supplied and passed)")
    disclaimer: str = _DISCLAIMER


class PredictInput(BaseModel):
    """Optional feature payload. Empty body -> the endpoint acts as a capability query (returns the
    envelope). A non-empty payload triggers the L0 gate, which ABSTAINS on malformed/OOD input."""
    features: dict[str, float] = Field(default_factory=dict)
    sample_id: str | None = None


def _l0_validate(features: dict[str, float]) -> list[str]:
    """L0 data-validation. Returns the reasons to ABSTAIN (empty list = passes). Never fabricates a number
    on bad input — the honest response to insufficient/OOD input is abstention, not a guess."""
    reasons: list[str] = []
    vals = list(features.values())
    if not any(math.isfinite(v) for v in vals):
        reasons.append("no finite feature values")
    if any(not math.isfinite(v) for v in vals):
        reasons.append("non-finite feature value(s) present (NaN/inf)")
    # crude OOD guard without a deployed reference model: standardized inputs should not be wildly scaled;
    # |value| > 1e6 is out-of-distribution for any real standardized/clinical feature we serve.
    if any(math.isfinite(v) and abs(v) > 1e6 for v in vals):
        reasons.append("feature magnitude out of distribution (|x| > 1e6)")
    return reasons


def _view(c) -> CapabilityView:
    return CapabilityView(
        name=c.name, layer=c.dashboard_layer, status=c.status.value, tier=c.tier.value,
        evidence=c.evidence, blocked_reason=c.blocked_reason,
        servable_now=(c.status != Status.BLOCKED),
    )


def _envelope(q: Question) -> ClaimEnvelope:
    caps = [_view(c) for c in capabilities_for(q)]
    return ClaimEnvelope(
        question=_SHORT[q],
        servable_now=[c.name for c in caps if c.servable_now],
        blocked=[c.name for c in caps if not c.servable_now],
        capabilities=caps,
    )


def create_honest_app() -> FastAPI:
    app = FastAPI(
        title="ResistanceMap — honest what/when/why surface",
        description="Serves only what open data supports; every response carries its claim-tier (goal.py).",
        version="1.0.0-honest",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "surface": "honest", "source_of_truth": "resistancemap.goal"}

    @app.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return {"summary": summary(),
                "questions": {_SHORT[q]: _envelope(q).model_dump() for q in Question},
                "disclaimer": _DISCLAIMER}

    @app.get("/capabilities/{question}", response_model=ClaimEnvelope)
    def capabilities_for_question(question: str) -> ClaimEnvelope:
        if question not in _QMAP:
            raise HTTPException(status_code=404, detail=f"unknown question '{question}' (what|when|why)")
        return _envelope(_QMAP[question])

    @app.post("/predict/trajectory")
    def predict_trajectory() -> dict[str, Any]:
        # the dynamic 3/6/12-mo trajectory is BLOCKED on open baseline-only data — refuse, never fabricate
        blocked = [c for c in capabilities_for(Question.WHEN) if c.status == Status.BLOCKED]
        reason = (blocked[0].blocked_reason if blocked and blocked[0].blocked_reason
                  else "dynamic trajectory needs a real serial substrate (T2+); open data is baseline-only.")
        raise HTTPException(status_code=422, detail={
            "refused": "predict/trajectory",
            "status": Status.BLOCKED.value,
            "blocked_reason": reason,
            "note": "Substrate gate FAILS on baseline-only open data (Result A). No trajectory fabricated.",
        })

    @app.post("/predict/{question}", response_model=ClaimEnvelope)
    def predict(question: str, inp: PredictInput | None = Body(default=None)) -> ClaimEnvelope:
        if question not in _QMAP:
            raise HTTPException(status_code=404, detail=f"unknown question '{question}' (what|when|why)")
        env = _envelope(_QMAP[question])
        if not env.servable_now:
            raise HTTPException(status_code=422, detail={
                "question": question, "status": Status.BLOCKED.value,
                "blocked": env.blocked, "note": "all capabilities for this question are blocked."})
        # L0 data-validation: only when a feature payload is supplied. Empty body = capability query.
        if inp is not None and inp.features:
            reasons = _l0_validate(inp.features)
            if reasons:
                raise HTTPException(status_code=422, detail={
                    "abstained": True, "question": question, "reasons": reasons,
                    "note": "L0 gate ABSTAINS on malformed/OOD input — no number is fabricated."})
            env.gate = {"l0_validation": "passed", "n_features": len(inp.features),
                        "claim_tier": [c.status for c in env.capabilities],
                        "run_level_gates": "release-time evidence gating lives in "
                                           "resistancemap.governance.claim_gates (RunEvidence-based)"}
        return env

    @app.get("/targets/{gene}")
    def targets(gene: str) -> dict[str, Any]:
        why = _envelope(Question.WHY)
        return {"gene": gene, "question": "why", "claim_tier": "ASSOCIATIONAL",
                "capabilities": [c.model_dump() for c in why.capabilities],
                "note": "Target attribution is ASSOCIATIONAL — from the STRING-graph null-rewiring harness "
                        "(real PPI beats rewired null on cross-fold reproducibility). Not causal.",
                "disclaimer": _DISCLAIMER}

    return app
