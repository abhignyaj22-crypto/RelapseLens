"""Tests for the Phase-3 falsification controls (permuted-omics, random-label)."""
import numpy as np
import pandas as pd

from mm_tte_survival.comparators import PenalizedCox
from mm_tte_survival.metrics_ipcw import uno_c_index
from mm_tte_survival.experiments_falsification import _fold_frame, _decide


def _cohort(seed=0, n=400):
    rng = np.random.default_rng(seed)
    om = rng.normal(size=n)                      # the only prognostic feature
    clin = rng.normal(size=n)                    # pure noise
    risk = 1.0 * om
    t = np.exp(-risk) * rng.weibull(1.3, n) * 20 + 1
    e = (rng.random(n) < 0.7).astype(int)
    df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)], "time_months": t, "event": e,
                       "clin": clin, "om": om})
    return df.iloc[:260].reset_index(drop=True), df.iloc[260:].reset_index(drop=True), rng


def _unoc(train, test, cols):
    comp = PenalizedCox(cols).fit(train)
    return uno_c_index(train.time_months.values, train.event.values,
                       test.time_months.values, test.event.values, comp.risk(test))


def test_permuted_omics_destroys_the_molecular_signal():
    tr, te, rng = _cohort()
    real = _unoc(tr, te, ["clin", "om"])
    # permute the omics column across patients in BOTH splits (break om<->outcome)
    tr_p = tr.assign(om=tr.om.values[rng.permutation(len(tr))])
    te_p = te.assign(om=te.om.values[rng.permutation(len(te))])
    permuted = _unoc(tr_p, te_p, ["clin", "om"])
    assert real > 0.62                       # real omics carries signal
    assert real - permuted > 0.08            # permuting omics collapses discrimination


def test_random_labels_give_chance_discrimination():
    tr, te, rng = _cohort()
    # shuffle (time,event) as a paired unit -> no learnable signal
    p = rng.permutation(len(tr))
    tr_r = tr.assign(time_months=tr.time_months.values[p], event=tr.event.values[p])
    c = _unoc(tr_r, te, ["clin", "om"])
    assert abs(c - 0.5) < 0.12               # ~ chance


def test_fold_frame_controls_actually_permute():
    n = 40
    ids = np.array([f"p{i}" for i in range(n)])
    df = pd.DataFrame({"patient_id": ids, "c1": np.arange(n, dtype=float),
                       "amp1q": (np.arange(n) % 2).astype(float)})
    genes = pd.DataFrame(np.random.default_rng(0).normal(size=(n, 50)),
                         index=[str(i) for i in ids], columns=[f"g{j}" for j in range(50)])
    frames = _fold_frame(df, genes, ["c1"], ["amp1q"], ids.astype(str), ids[:30].astype(str),
                         k=4, n_topvar=20, seed=1, rng=np.random.default_rng(2))
    om = frames["om_cols"]
    # permuted-omics differs from real in the omics block but shares the clinical column
    assert not np.allclose(frames["real"][om].values, frames["perm_omics"][om].values)
    assert np.allclose(frames["real"]["c1"].values, frames["perm_omics"]["c1"].values)
    # scrambled-subtype differs in the cyto column
    assert not np.array_equal(frames["real"]["amp1q"].values, frames["scram_cyto"]["amp1q"].values)


def test_decide_returns_H0_when_nothing_significant():
    lb = pd.DataFrame({"arm": ["clinical", "clinical+omics", "permuted_omics", "random_label"],
                       "mean_uno_cindex": [0.70, 0.71, 0.705, 0.50]})
    deltas = pd.DataFrame({
        "comparison": ["clinical+omics_vs_clinical"],
        "delta_cindex": [0.01], "delta_ci_low": [-0.03], "p_permutation": [0.4],
        "p_fdr_bh": [0.4], "fdr_significant": [False]})
    dec = _decide(lb, deltas)
    assert "H0 retained" in dec["overall"]
    assert "NULL" in dec["molecular_increment"]["claim"]
