"""Tests for the leak-free in-fold omics PCA and the gene-matrix cache."""
import numpy as np
import pandas as pd
import pytest

from mm_tte_survival.data.omics_pca import OmicsInFoldPCA


def _toy_genes(n=60, g=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"p{i}" for i in range(n)]
    return pd.DataFrame(rng.normal(size=(n, g)), index=ids,
                        columns=[f"g{j}" for j in range(g)])


def test_infold_pca_shapes_and_transform():
    gene = _toy_genes()
    train = gene.index[:40]
    pca = OmicsInFoldPCA(k=8, n_topvar=50, seed=0).fit(gene, train)
    assert len(pca.top_genes_) == 50
    out = pca.transform(gene)
    assert out.shape == (60, 8)
    assert list(out.columns) == [f"PC{i+1}" for i in range(8)]


def test_infold_pca_is_fit_on_train_only():
    # different train sets -> different selected genes / components (leak-free signal)
    gene = _toy_genes()
    a = OmicsInFoldPCA(k=5, n_topvar=30, seed=0).fit(gene, gene.index[:30])
    b = OmicsInFoldPCA(k=5, n_topvar=30, seed=0).fit(gene, gene.index[30:])
    # the top-variable gene set depends on which rows are train
    assert set(a.top_genes_) != set(b.top_genes_)


def test_gene_matrix_roundtrip(tmp_path):
    from mm_tte_survival.data.gene_expression import load_gene_matrix
    p = tmp_path / "gm.npz"
    np.savez_compressed(p, matrix=np.arange(6, dtype="float32").reshape(2, 3),
                        patient_ids=np.array(["a", "b"], dtype=object),
                        genes=np.array(["g1", "g2", "g3"], dtype=object))
    df = load_gene_matrix(p)
    assert df.shape == (2, 3) and list(df.index) == ["a", "b"]
    assert df.loc["b", "g3"] == 5.0
