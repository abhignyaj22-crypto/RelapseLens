"""A1 2.4 — the fusion harness honestly tests the dashboard's 0.82->0.89 multimodal claim.

Pins the contract, not the ~90s real-data run: fusion must beat the BEST SINGLE modality (not merely the
per-drug mean) with a grouped-bootstrap CI excluding 0 — otherwise NULL. It reuses the 2.2 GDSC2×CCLE join
and adds DepMap CRISPR essentiality as the second modality, keeping deepcdr fail-loud. Measured verdict:
NULL — fusion − best-single = +0.002 [−0.003, +0.007]; the multimodal lift does not replicate (see
docs/autoresearch/SCORECARD.md). Real data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_AR = ROOT / "scripts" / "autoresearch"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))


def test_fusion_gate_is_beat_best_single_with_ci():
    src = (_AR / "run_fusion_gdsc2.py").read_text()
    # the honest bar: fusion vs the BEST single modality, not just per-drug-mean
    assert "best_single" in src and "fusion_minus_best_single_ci" in src
    assert "expr" in src and "crispr" in src.lower()          # two real modalities fused
    assert "deepcdr_reimpl" in src                            # SOTA row stays fail-loud


def test_fusion_reuses_the_2_2_join():
    src = (_AR / "run_fusion_gdsc2.py").read_text()
    assert "from run_drug_gdsc2 import" in src                # reuses the 2.2 GDSC2×CCLE join, no duplication


def test_fusion_harness_imports():
    import importlib
    m = importlib.import_module("run_fusion_gdsc2")
    assert hasattr(m, "run") and hasattr(m, "main")
