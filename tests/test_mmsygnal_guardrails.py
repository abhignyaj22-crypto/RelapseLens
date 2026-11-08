"""Guardrails for the mmSYGNAL external benchmark (GPL-3.0, not vendored)."""
from pathlib import Path
import subprocess
import yaml

REPO = Path(__file__).resolve().parents[1]
RDS_NAMES = [
    "agnostic_risk_model.Rds", "amp(1q)_risk_model.Rds", "del(13)_risk_model.Rds",
    "del(1p)_risk_model.Rds", "t(4;14)_risk_model.Rds", "FGFR3_risk_model.Rds",
]


def test_mmsygnal_weights_not_vendored():
    """No mmSYGNAL .Rds artifact may live OUTSIDE external/ (would carry GPL-3.0)."""
    offenders = [str(p.relative_to(REPO)) for p in REPO.rglob("*.Rds")
                 if "external/" not in str(p.relative_to(REPO))]
    assert not offenders, f"mmSYGNAL/.Rds artifacts copied outside external/: {offenders}"


def test_mmsygnal_not_git_tracked():
    """Git must not track the cloned external repo or its weights."""
    try:
        tracked = subprocess.check_output(
            ["git", "-C", str(REPO), "ls-files", "external/"], text=True).strip()
    except Exception:
        tracked = ""
    assert "mmSYGNAL" not in tracked, f"external mmSYGNAL is git-tracked: {tracked[:200]}"


def test_benchmark_claim_policy_blocks_relapse():
    cfg = yaml.safe_load((REPO / "configs/benchmarks/mmsygnal.yaml").read_text())
    pol = cfg["claim_policy"]
    assert pol["direct_relapse_pfs_claim_allowed"] is False
    assert pol["clinical_use_allowed"] is False
    assert pol["research_benchmark_allowed"] is True


def test_provenance_doc_records_license_and_commit():
    doc = (REPO / "docs/third_party/mmSYGNAL_provenance.md").read_text()
    assert "GPL-3.0" in doc
    assert "6017dd8d" in doc  # upstream commit


# ---- Step 2/3: schema validator must reject invalid inputs ----------------- #
import pandas as pd
import pytest
from mm_tte_survival.benchmarks.mmsygnal_schema import (
    validate_mmsygnal_program_activity, is_valid_mmsygnal_program_activity)


def test_rejects_rna_pcs(tmp_path):
    """RNA PCs (PC1..PC128) are NOT mmSYGNAL program activity -> reject."""
    df = pd.DataFrame({"patient_id": ["A", "B"], **{f"PC{i}": [0.1, 0.2] for i in range(1, 30)}})
    p = tmp_path / "pcs.csv"; df.to_csv(p, index=False)
    assert not is_valid_mmsygnal_program_activity(p)
    with pytest.raises(ValueError, match="141 program activity columns"):
        validate_mmsygnal_program_activity(p)


def test_rejects_curated_signature_programs(tmp_path):
    """The repo's 10 curated prog_* signatures are NOT the 141 mmSYGNAL programs."""
    df = pd.DataFrame({"patient_id": ["A", "B"],
                       "prog_proliferation": [0.1, 0.2], "prog_mmset_t4_14": [0.0, 0.5],
                       "prog_maf_program": [0.3, 0.1]})
    p = tmp_path / "sigs.csv"; df.to_csv(p, index=False)
    assert not is_valid_mmsygnal_program_activity(p)
    with pytest.raises(ValueError):
        validate_mmsygnal_program_activity(p)


def test_accepts_valid_0_140_and_prefixed(tmp_path):
    raw = pd.DataFrame({"patient_id": ["A"], **{str(i): [0.0] for i in range(141)}})
    p = tmp_path / "raw.csv"; raw.to_csv(p, index=False)
    out = validate_mmsygnal_program_activity(p)
    assert list(out.columns) == ["patient_id"] + [str(i) for i in range(141)]
    pref = pd.DataFrame({"patient_id": ["A"], **{f"program_{i}": [0.0] for i in range(141)}})
    p2 = tmp_path / "pref.csv"; pref.to_csv(p2, index=False)
    assert is_valid_mmsygnal_program_activity(p2)


def test_real_curated_program_activity_is_rejected():
    """The actual data/real/program_activity.csv (if present) must be rejected."""
    f = REPO / "data/real/program_activity.csv"
    if f.exists():
        assert not is_valid_mmsygnal_program_activity(f)


def test_build_scaffold_fails_closed():
    """The program-activity builder must FAIL rather than approximate."""
    txt = (REPO / "scripts/benchmarks/build_mmsygnal_program_activity.R").read_text()
    assert 'stop(' in txt and "Refusing to emit approximate" in txt
