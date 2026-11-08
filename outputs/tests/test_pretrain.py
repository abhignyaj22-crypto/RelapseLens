"""Quick-win: self-supervised masked-gene pretraining of the transcriptome encoder.

On data with real gene-gene correlation structure, masked reconstruction must beat the naive
(predict-mean) baseline — i.e. held-out masked R² > 0 — and the reconstruction loss must fall.
Verified on a small correlated synthetic matrix (this is a UNIT test of the LEARNING mechanism,
not a data-backed result; the real run uses CoMMpass RNA).
"""
from __future__ import annotations

import numpy as np

from mm_tte_survival.models.encoders import TranscriptomeEncoder
from mm_tte_survival.training.pretrain import masked_gene_pretrain


def _correlated_matrix(n=200, g=40, k=5, seed=0):
    """n×g expression with low-rank (k) gene-gene structure so masked inference is possible."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(k, g))
    scores = rng.normal(size=(n, k))
    return scores @ loadings + 0.1 * rng.normal(size=(n, g))


def test_masked_pretrain_learns_gene_structure():
    X = _correlated_matrix()
    res = masked_gene_pretrain(X, latent_dim=16, hidden=64, epochs=250, device="cpu")
    assert res.masked_r2 > 0.1                     # beats predict-mean on masked entries
    assert res.history[-1] < res.history[0]        # reconstruction loss fell
    assert isinstance(res.encoder, TranscriptomeEncoder)


def test_pretrained_encoder_is_usable():
    import torch
    X = _correlated_matrix(n=64, g=20)
    res = masked_gene_pretrain(X, latent_dim=8, hidden=32, epochs=50, device="cpu")
    z = res.encoder(torch.tensor(X[:4], dtype=torch.float32))
    assert z.shape == (4, 8) and torch.isfinite(z).all()
