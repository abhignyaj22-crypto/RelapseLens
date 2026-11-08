"""The dashboard goal → code → honest status registry (A1 refactor spine).

Single source of truth mapping the product goal — *"predict which drugs will fail, WHEN they will fail,
and WHY"* (abhignya-jagathpally.github.io/dashboard/#/science) — onto this repo's modules, the dashboard's
L0–L9 layer stack, the karpathy/autoresearch keep-revert entrypoint that governs each claim, and each
claim's honest data-availability tier.

This module is DECLARATIVE ONLY (dataclasses + strings) — it imports nothing heavy and pulls no torch, so
it is cheap and safe to import and cannot create an acyclic-dependency violation. It is the map the A1
refactor organizes around: every capability is one of three QUESTIONS, driven by a fixed-prepare /
pre-registered keep-revert loop, and gated to what REAL open data supports today. Advertised-but-unsupported
layers are marked BLOCKED with the tier that would unlock them — never fabricated, never softened.

Tiers T1–T5 mirror `docs/paper/progression_potential_design.md` §3.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Question(str, Enum):
    WHAT = "what_fails"          # which drugs will fail — per-drug resistance probability
    WHEN = "when_progression"    # when they will fail — time-to-event / horizon risk
    WHY = "why_biological"       # why — pathway / graph / mechanistic attribution


class Status(str, Enum):
    ANSWERABLE = "answerable_now"        # REAL open data supports this claim today
    PARTIAL = "partial"                  # supported with honest caveats / modest effect
    BLOCKED = "blocked_pending_data"     # needs controlled / serial data not on the open tier


class Tier(str, Enum):
    T1 = "T1_baseline_os"                # open GDC MMRF baseline + OS — ON DISK
    T2 = "T2_serial_clinical"            # per-visit labs + regimen dates — controlled
    T3 = "T3_baseline_molec_serial"      # T2 + baseline omics — controlled
    T4 = "T4_paired_molecular"           # ≥2 real molecular timepoints — absent
    T5 = "T5_dense_serial_molecular"     # dense molecular timeseries — absent


@dataclass(frozen=True)
class Capability:
    question: Question
    name: str
    dashboard_layer: str                 # L0..L9
    modules: tuple[str, ...]             # repo paths (relative) that implement it
    autoresearch_entry: str | None       # the fixed-prepare / keep-revert runner, if wired
    status: Status
    tier: Tier
    evidence: str                        # honest one-line current evidence
    blocked_reason: str = ""


# --- dashboard layer stack → representative module (honest mapping, honest-core edition) ---
# After the honest-core cleanup the aspirational breadth layers (L1 VAE manifold, L3 GAT, L5 Waddington
# landscape, L7 pathway attribution) are no longer implemented in this repo — the dashboard advertised them
# but no real open-data-backed code delivered them. LAYERS now maps ONLY the layers backed by surviving,
# tested code (the trusted survival/eval core + the honest servable seam). Each entry points to a real module.
LAYERS: dict[str, tuple[str, str]] = {
    "L0": ("data validation + provenance", "src/mm_tte_survival/data"),
    "L2": ("time-to-event survival core (OS benchmark + defective-FHT)",
           "src/mm_tte_survival/experiments_falsification.py"),
    "L4": ("multi-modal fusion (baseline arms)", "src/mm_tte_survival/models/fusion.py"),
    "L6": ("governance / claim-gate", "src/mm_tte_survival/evaluation/claim_gate.py"),
    "L8": ("conformal / uncertainty abstention", "src/mm_tte_survival/conformal_survival.py"),
    "L9": ("serve (honest FastAPI /predict — claim-tier per response)", "src/resistancemap/inference/honest_api.py"),
}


CAPABILITIES: tuple[Capability, ...] = (
    # ---------------- WHEN (time-to-event) ----------------
    Capability(
        Question.WHEN, "static OS survival benchmark", "L2",
        ("src/mm_tte_survival/experiments_falsification.py", "src/mm_tte_survival/training/losses.py"),
        "scripts/autoresearch/run_experiment.py", Status.ANSWERABLE, Tier.T1,
        "leak-proof OS benchmark; flat penalised Cox best (Uno-C 0.7546); rank ceiling falsified 6 ways."),
    Capability(
        Question.WHEN, "defective first-hitting-time head", "L2",
        ("src/mm_tte_survival/longitudinal/crossing_head.py", "scripts/paper/result_set_C_fht_ablation.py"),
        "scripts/paper/result_set_C_fht_ablation.py", Status.ANSWERABLE, Tier.T1,
        "Result C: defective-FHT repairs proper-FHT calibration (IBS 0.328→0.140, D-cal 0.000→0.809)."),
    Capability(
        Question.WHEN, "conformal calibrated survival bound", "L8",
        ("src/mm_tte_survival/conformal_survival.py",), "scripts/autoresearch/run_conformal_e3.py",
        Status.PARTIAL, Tier.T1,
        "E3: marginal coverage repaired 0.58→0.87; per-stratum (Mondrian) NULL on rare subtypes."),
    Capability(
        Question.WHEN, "dynamic trajectory forecast (3/6/12mo)", "L2",
        ("src/mm_tte_survival/models/clinical_teacher.py", "src/mm_tte_survival/data/serial_clinical.py"),
        None, Status.BLOCKED, Tier.T2,
        "A1 2.5: the WHEN core shares ONE substrate gate — ClinicalTeacher.assert_trainable_on calls "
        "require_real_serial_substrate and refuses the real baseline-only data (tested).",
        blocked_reason="open GDC is baseline-only (995 pts × 1 visit) — Result A substrate gate FAILS."),
    # ---------------- WHAT (per-drug resistance) ----------------
    Capability(
        Question.WHAT, "per-drug response (bortezomib)", "L4",
        ("scripts/autoresearch/run_drug_e2.py",),
        "scripts/autoresearch/run_drug_e2.py", Status.PARTIAL, Tier.T1,
        "E2: GSE9782 bortezomib AUROC 0.589 [0.518, 0.661] (permuted 0.451) — modest, real."),
    Capability(
        Question.WHAT, "multimodal per-(sample,drug) fusion embedding", "L4",
        ("src/mm_tte_survival/models/fusion.py", "scripts/autoresearch/run_fusion_gdsc2.py"),
        "scripts/autoresearch/run_fusion_gdsc2.py", Status.PARTIAL, Tier.T1,
        "cell-line RNA/CNV per-drug on DepMap/GDSC2; the advertised fusion 0.82→0.89 lift does NOT replicate "
        "on open data (run_fusion_gdsc2 NULL) — baseline arms tie."),
    # ---------------- WHY (mechanism) ----------------
    Capability(
        Question.WHY, "molecular incremental value (RQ3)", "L7",
        ("src/mm_tte_survival/experiments_falsification.py",), "scripts/autoresearch/run_experiment.py",
        Status.ANSWERABLE, Tier.T1,
        "H3 NULL retained on OS: no molecular arm beats clinical; permuted-omics control holds."),
    Capability(
        Question.WHY, "program / pathway attribution (do-program)", "L7",
        ("scripts/autoresearch/run_attribution_e5.py",),
        "scripts/autoresearch/run_attribution_e5.py", Status.PARTIAL, Tier.T1,
        "E5 ASSOCIATIONAL: proliferation p=0.010, high-risk-GEP p=0.012; causal channels not evaluable."),
    Capability(
        Question.WHY, "STRING-PPI graph smoothing (null-rewiring validated)", "L3",
        ("scripts/autoresearch/run_why_graph.py",),
        "scripts/autoresearch/run_why_graph.py", Status.PARTIAL, Tier.T1,
        "A1 2.3 ASSOCIATIONAL: real STRING PPI beats the degree-preserving rewired null on cross-fold "
        "top-100 Jaccard (0.172 vs [0.058,0.084]; no-smoothing 0.027) — reproducible risk module, modest."),
)


# --------------------------------------------------------------------------- helpers
def capabilities_for(question: Question) -> tuple[Capability, ...]:
    return tuple(c for c in CAPABILITIES if c.question == question)


def by_status(status: Status) -> tuple[Capability, ...]:
    return tuple(c for c in CAPABILITIES if c.status == status)


def summary() -> dict[str, dict[str, int]]:
    """Counts per question × status — the honest scoreboard of what's real vs blocked."""
    out: dict[str, dict[str, int]] = {}
    for q in Question:
        out[q.value] = {s.value: len([c for c in capabilities_for(q) if c.status == s]) for s in Status}
    return out
