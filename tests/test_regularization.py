"""Tests for the Direction-2 regularization-probe decision logic."""
import numpy as np
import pandas as pd

from mm_tte_survival.experiments_regularization import _decide


def _summ(neural_ibs, cox_ibs):
    rows = []
    for s in ["del17p", "del1p", "amp1q"]:
        rows.append({"subtype": s, "model": "pooled_neural", "mean_ibs": neural_ibs})
        rows.append({"subtype": s, "model": "pooled_cox", "mean_ibs": cox_ibs})
    return pd.DataFrame(rows)


def _M():
    M = np.zeros((100, 3), dtype="float32")
    M[:10, 0] = 1; M[:20, 1] = 1; M[:50, 2] = 1   # del17p smallest, then del1p
    return M


def test_null_when_neural_not_better():
    d = _decide(_summ(neural_ibs=0.17, cox_ibs=0.158), _M(), ["del17p", "del1p", "amp1q"])
    assert d["neural_beats_cox"] is False and "NULL" in d["verdict"]


def test_neural_helps_when_clearly_lower_ibs():
    d = _decide(_summ(neural_ibs=0.12, cox_ibs=0.158), _M(), ["del17p", "del1p", "amp1q"])
    assert d["neural_beats_cox"] is True and "helps" in d["verdict"].lower()
