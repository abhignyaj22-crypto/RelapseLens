"""sota-workaround A1 — drug fingerprint featurization (rdkit ECFP4).

Unit-tests the pure fingerprint function on known SMILES (no network / no fetched data). The SMILES cache +
fingerprint matrix are gitignored, rebuildable real PubChem artifacts (scripts/autoresearch/fetch_gdsc2_smiles.py);
an optional check validates them if present. Real data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")
ROOT = Path(__file__).resolve().parents[1]
_AR = ROOT / "scripts" / "autoresearch"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))
import fetch_gdsc2_smiles as F  # noqa: E402


def test_morgan_shape_and_bits():
    fp = F.morgan("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
    assert fp.shape == (F.N_BITS,) and fp.dtype == np.int8
    assert 0 < fp.sum() < F.N_BITS           # some substructure bits set, not all


def test_morgan_distinguishes_molecules():
    aspirin = F.morgan("CC(=O)Oc1ccccc1C(=O)O")
    caffeine = F.morgan("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    assert not np.array_equal(aspirin, caffeine)   # different chemistry -> different fingerprint


def test_fetched_artifact_is_real_if_present():
    csv = ROOT / "data/external/drug_response/gdsc2_drug_smiles.csv"
    if not csv.exists():
        pytest.skip("SMILES cache not fetched (gitignored, rebuildable via fetch_gdsc2_smiles.py)")
    import pandas as pd
    from rdkit import Chem
    df = pd.read_csv(csv)
    resolved = df[df["smiles"].notna()]
    assert len(resolved) >= 50                                  # a usable drug panel
    assert all(Chem.MolFromSmiles(s) is not None for s in resolved["smiles"].head(20))  # rdkit-valid
