"""Static leak-audit over the Stage-D/E/F ResistanceMap scripts (review: extend
_audit_leakage coverage to the stage scripts). These scripts were scaffolded outside
the canonical evaluate_model_suite (which already hard-raises on full-cohort PC* and
test-row-in-fit). This regression guard freezes the leak-free routing so the leaky
legacy substrate can never creep back in:

  1. No stage script may READ the full-cohort omics.csv as a feature source — the
     leak-free path is gene_matrix.npz + OmicsInFoldPCA (fit train-only per split).
  2. Every stage_e script that builds transcriptomic features must reference
     OmicsInFoldPCA (i.e. it actually uses the in-fold transformer).

The check is AST-based so docstrings / comments that merely *mention* omics.csv (to say
it is NOT used) do not trip it — only real read_csv(...) calls do.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE_DIRS = [ROOT / "scripts/stage_d", ROOT / "scripts/stage_e", ROOT / "scripts/stage_f"]
LEAKY_FEATURE_FILES = {"omics.csv"}  # full-cohort PCs — membership/EDA only, never features


def _stage_scripts():
    return [p for d in STAGE_DIRS if d.exists() for p in sorted(d.glob("*.py"))]


def _string_constants_in_call_args(call: ast.Call):
    for arg in ast.walk(call):
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            yield arg.value


def _read_csv_targets(tree: ast.AST):
    """Yield the string literals passed to any read_csv(...) call (incl. Path / "x.csv")."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == "read_csv":
                yield from _string_constants_in_call_args(node)


@pytest.mark.parametrize("script", _stage_scripts(), ids=lambda p: str(p.relative_to(ROOT)))
def test_stage_script_does_not_read_leaky_omics_as_features(script):
    tree = ast.parse(script.read_text())
    reads = {t for t in _read_csv_targets(tree)}
    leaked = {t for t in reads if any(t.endswith(f) or f in t for f in LEAKY_FEATURE_FILES)}
    assert not leaked, (
        f"{script.relative_to(ROOT)} reads leaky full-cohort file(s) {leaked} via read_csv; "
        f"route transcriptomic features through gene_matrix.npz + OmicsInFoldPCA (in-fold).")


def test_fusion_script_uses_infold_pca():
    """The proposed multi-modal model must fit its transcriptome PCA in-fold."""
    fusion = ROOT / "scripts/stage_e/train_fusion_model.py"
    if not fusion.exists():
        pytest.skip("train_fusion_model.py not present")
    assert "OmicsInFoldPCA" in fusion.read_text(), (
        "train_fusion_model.py must use OmicsInFoldPCA for leak-free transcriptome features")
