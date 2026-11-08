"""The two C-index implementations (reference + vectorised) must agree exactly."""
import numpy as np
from mm_tte_survival.metrics import harrell_c_index, fast_c_index


def test_cindex_implementations_agree():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(20, 120)
        t = rng.uniform(1, 60, n)
        e = rng.integers(0, 2, n)
        r = rng.normal(size=n)
        a = harrell_c_index(t, e, r)
        b = fast_c_index(t, e, r)
        if np.isfinite(a) and np.isfinite(b):
            assert abs(a - b) < 1e-9, f"c-index mismatch {a} vs {b}"
