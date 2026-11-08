"""Quick-win: self-supervised masked-gene pretraining for the transcriptome encoder.

Masked-gene autoencoding (BERT-style for expression): randomly zero a fraction of input genes,
encode, and reconstruct the MASKED entries. This is LABEL-FREE — it never touches survival
labels — so pretraining on the full CoMMpass RNA cohort is not survival leakage; it just gives
the encoder a better initialisation before survival fine-tuning.

Scope note: the survival path currently feeds the encoder in-fold PCA(16) inputs, so to USE a
pretrained encoder end-to-end you instantiate a raw-gene encoder (input_dim = n_genes) here and
carry its weights into a raw-gene survival variant. This module provides that capability + is
verified (reconstruction improves, held-out masked R² > 0); wiring it into a measured ΔC run is
the documented integration step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..models.encoders import TranscriptomeEncoder


class MaskedGeneAutoencoder(nn.Module):
    """TranscriptomeEncoder + linear decoder, trained by masked reconstruction."""

    def __init__(self, n_genes: int, latent_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.encoder = TranscriptomeEncoder(n_genes, latent_dim=latent_dim, hidden=hidden)
        self.decoder = nn.Linear(latent_dim, n_genes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass
class PretrainResult:
    encoder: TranscriptomeEncoder
    history: list
    masked_r2: float           # held-out reconstruction R² on masked entries (sanity > 0)


def masked_gene_pretrain(X: np.ndarray, latent_dim: int = 64, hidden: int = 256,
                         mask_frac: float = 0.25, epochs: int = 200, lr: float = 1e-3,
                         seed: int = 42, device: str | None = None) -> PretrainResult:
    """Pretrain a TranscriptomeEncoder on X (n_samples × n_genes) by masked reconstruction.

    Returns the pretrained encoder + the held-out masked R² (a positive value means the encoder
    learned real gene-gene structure, not identity). No labels used anywhere.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype="float32")
    n, g = X.shape
    idx = rng.permutation(n)
    n_val = max(8, n // 10)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    # standardize per gene on TRAIN only (self-supervised, but keep the split clean)
    mu, sd = X[tr_i].mean(0), X[tr_i].std(0) + 1e-6
    Xz = torch.tensor((X - mu) / sd, device=dev)
    Xtr, Xva = Xz[tr_i], Xz[val_i]

    model = MaskedGeneAutoencoder(g, latent_dim=latent_dim, hidden=hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        mask = (torch.rand_like(Xtr) < mask_frac).float()        # 1 = masked (hidden from input)
        recon = model(Xtr * (1.0 - mask))
        # loss only on masked entries — force gene-gene inference, not copy-through
        denom = mask.sum().clamp_min(1.0)
        loss = ((recon - Xtr) ** 2 * mask).sum() / denom
        loss.backward(); opt.step()
        history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        vmask = (torch.rand_like(Xva) < mask_frac).float()
        vrec = model(Xva * (1.0 - vmask))
        m = vmask > 0.5
        y, yh = Xva[m], vrec[m]
        ss_res = float(((y - yh) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-8
        masked_r2 = 1.0 - ss_res / ss_tot
    return PretrainResult(encoder=model.encoder.to("cpu"), history=history, masked_r2=masked_r2)


__all__ = ["MaskedGeneAutoencoder", "masked_gene_pretrain", "PretrainResult"]
