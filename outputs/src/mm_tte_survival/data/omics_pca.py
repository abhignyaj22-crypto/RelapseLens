"""Leak-free, in-fold omics PCA.

Top-variable-gene selection, standardisation, and PCA are ALL fit on the train
fold only, then applied to every row. This is the reusable transformer that both
the Stage-A re-baseline and (later) evaluate_model_suite call identically — so
adopting it on the main path is a drop-in, not a rewrite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class OmicsInFoldPCA:
    def __init__(self, k: int = 16, n_topvar: int = 2000, seed: int = 42):
        self.k = k
        self.n_topvar = n_topvar
        self.seed = seed
        self.top_genes_: list[str] | None = None
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.n_components_: int | None = None

    def fit(self, gene_df: pd.DataFrame, train_ids) -> "OmicsInFoldPCA":
        train_ids = [str(p) for p in train_ids if str(p) in set(gene_df.index)]
        Xtr = gene_df.loc[train_ids]
        if len(Xtr) < 3:
            raise ValueError("too few train patients with gene expression for in-fold PCA")
        # top-variable genes — variance computed on TRAIN only
        top = Xtr.var(axis=0).sort_values(ascending=False).index[:self.n_topvar]
        self.top_genes_ = list(top)
        self.scaler_ = StandardScaler().fit(Xtr[self.top_genes_].values)
        n_comp = int(min(self.k, len(self.top_genes_), len(Xtr) - 1))
        self.pca_ = PCA(n_components=n_comp, random_state=self.seed).fit(
            self.scaler_.transform(Xtr[self.top_genes_].values))
        self.n_components_ = n_comp
        return self

    def transform(self, gene_df: pd.DataFrame) -> pd.DataFrame:
        if self.pca_ is None:
            raise RuntimeError("call fit() before transform()")
        Z = self.pca_.transform(self.scaler_.transform(gene_df[self.top_genes_].values))
        cols = [f"PC{i+1}" for i in range(self.n_components_)]
        return pd.DataFrame(Z, index=gene_df.index, columns=cols)

    def fit_transform(self, gene_df: pd.DataFrame, train_ids) -> pd.DataFrame:
        return self.fit(gene_df, train_ids).transform(gene_df)
