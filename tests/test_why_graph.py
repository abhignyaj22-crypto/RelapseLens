"""A1 2.3 — the graph-WHY harness: null-rewiring falsification math + honest gating.

Unit-tests the pure functions (Jaccard stability, graph smoothing) on fixed fixtures and pins the harness
contract: it must use the rewiring null + STRING, keep everything ASSOCIATIONAL, and keep GraphMASK BLOCKED
(the layer exists but is not wired to a winning survival model — not faked). Measured verdict: CAPABILITY
(associational) — real-PPI cross-fold top-100 Jaccard 0.172 beats the rewired null [0.058,0.084] and
no-smoothing 0.027 (see docs/autoresearch/SCORECARD.md). The ~30s real-data run is not executed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
_AR = ROOT / "scripts" / "autoresearch"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))
import run_why_graph as W  # noqa: E402


def test_stability_jaccard_known_values():
    a = np.array([9, 8, 7, 1, 2])       # top-2 -> {0,1}
    b = np.array([9, 8, 1, 2, 7])       # top-2 -> {0,1}  identical
    c = np.array([1, 2, 9, 8, 7])       # top-2 -> {2,3}  disjoint
    assert W._stability([a, b], 2) == 1.0
    assert W._stability([a, c], 2) == 0.0
    # 3-way: pair(ab)=1, pair(ac)=0, pair(bc)=0 -> mean 1/3
    assert abs(W._stability([a, b, c], 2) - 1 / 3) < 1e-9


def test_smooth_identity_is_noop():
    a = np.array([1.0, -2.0, 3.0])
    ident = sp.identity(3, format="csr")
    assert np.allclose(W._smooth(a, ident), a)   # (1-α)a + α·I a == a


def test_harness_uses_rewiring_null_and_gates_graphmask():
    src = (_AR / "run_why_graph.py").read_text()
    assert "string_graph" in src.lower()                     # real STRING PPI
    assert "permutation" in src and "rewir" in src.lower()   # degree-preserving node-relabel null
    assert "ASSOCIATIONAL" in src                            # never a causal claim
    assert "BLOCKED" in src and "use_graphmask=False" in src # GraphMASK honestly gated, not faked
