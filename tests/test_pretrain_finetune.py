"""Wire-up test for the raw-gene pretrain → survival ΔC ablation (scripts/stage_e/pretrain_ablation.py).

This is a UNIT test of the MECHANISM (the pretrained encoder loads into the raw-gene survival
path and the paired-ΔC runner returns finite numbers), not a data-backed result — it runs on a
small correlated synthetic matrix, exactly like test_pretrain.py. The real ΔC comes from the
script on CoMMpass RNA and is a reportable NULL-or-not.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pretrain_ablation", ROOT / "scripts/stage_e/pretrain_ablation.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)


def _correlated(n=160, g=40, k=5, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n, k))
    return scores @ rng.normal(size=(k, g)) + 0.1 * rng.normal(size=(n, g)), scores


def test_pretrained_encoder_loads_into_survival_path():
    """A masked-gene-pretrained TranscriptomeEncoder must load into the fit/eval survival head
    and produce finite per-subject risk."""
    from mm_tte_survival.training.pretrain import masked_gene_pretrain
    X, _ = _correlated()
    pre = masked_gene_pretrain(X, latent_dim=pa.LAT, hidden=pa.HIDDEN, epochs=30, device="cpu")
    n = X.shape[0]
    tr, va, te = np.arange(0, 100), np.arange(100, 130), np.arange(130, n)
    t = np.abs(np.random.default_rng(1).normal(20, 5, n)) + 1
    e = (np.random.default_rng(2).random(n) > 0.4).astype(int)
    Xz = pa._standardize(X[tr], X)
    state = {k: v for k, v in pre.encoder.state_dict().items()}
    risk = pa.fit_eval_transcriptome(Xz[tr], t[tr], e[tr], Xz[va], t[va], e[va], Xz[te],
                                     input_dim=X.shape[1], hidden=pa.HIDDEN,
                                     pretrained_state=state, epochs=20, device="cpu")
    assert risk.shape == (len(te),) and np.isfinite(risk).all()


def test_run_ablation_returns_finite_paired_delta():
    """The paired-ΔC runner returns a finite ΔC + CI and a sane per-arm C on a small run."""
    X, scores = _correlated(n=180, g=30)
    ids = [f"p{i}" for i in range(len(X))]
    gene_df = pd.DataFrame(X, index=ids, columns=[f"g{j}" for j in range(X.shape[1])])
    # make survival depend weakly on latent structure so C is defined (not the reported result)
    risk_lat = scores[:, 0]
    t = np.clip(30 - 4 * risk_lat + np.random.default_rng(3).normal(0, 3, len(X)), 1, None)
    e = (np.random.default_rng(4).random(len(X)) > 0.4).astype(int)
    surv = pd.DataFrame({"patient_id": ids, "time_months": t, "event": e})
    res = pa.run_ablation(gene_df, surv, splits=2, pretrain_epochs=25,
                          with_pca_arm=True, device="cpu")
    ci = res["dC_pretrained_minus_random_mean_ci"]
    assert ci is not None and all(np.isfinite(ci))
    assert np.isfinite(res["C_pretrained_mean"]) and np.isfinite(res["C_random_mean"])
    assert res["masked_r2_mean"] == res["masked_r2_mean"]        # not NaN
    assert isinstance(res["gate_pretrain_helps"], bool)
