"""Stage D — cell-line multi-omic foundation (L1 VAE) + PPI graph encoder (L3).

Public cell-line/network data only (CCLE proteomics + histone-PTM, STRING v12 + ESM-2,
DepMap CRISPR / GDSC). These are the ResistanceMap representation layers, pretrained
here and transferred to patients in Stage E through the shared gene space + PPI graph.

NOT imported by ``models/__init__`` — kept out of the tested import surface until the
Stage-D deps (torch, torch_geometric, transformers) are installed, so the classical
120-test pipeline is unaffected. ``torch_geometric`` is imported lazily inside the GNN
so this module imports fine (for the VAE half) even before it is installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import ModalityEncoder


# --------------------------------------------------------------------------- #
# L1 — cell-line proteome -> epigenome VAE
# --------------------------------------------------------------------------- #
@dataclass
class VAEOutput:
    z: Tensor          # latent (reparameterized)
    mu: Tensor
    logvar: Tensor
    recon: Tensor      # reconstructed epigenome (target modality)


class CellLineVAE(ModalityEncoder):
    """VAE that encodes cell-line proteomics and reconstructs the paired epigenome
    (histone-PTM) — a shared multi-omic latent grounded in real CCLE measurements.

    Encoder: proteome -> (mu, logvar) -> z. Decoder: z -> epigenome. The KL + recon
    ELBO is trained on paired cell lines; `forward` returns the mean latent (mu) so it
    plugs into the uniform :class:`ModalityEncoder` contract used by fusion."""

    def __init__(self, proteome_dim: int, epigenome_dim: int, latent_dim: int = 64,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(proteome_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, epigenome_dim),
        )

    def encode(self, proteome: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(proteome)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        return mu + std * torch.randn_like(std)

    def vae_forward(self, proteome: Tensor) -> VAEOutput:
        mu, logvar = self.encode(proteome)
        z = self.reparameterize(mu, logvar)
        return VAEOutput(z=z, mu=mu, logvar=logvar, recon=self.decoder(z))

    def forward(self, proteome: Tensor) -> Tensor:  # ModalityEncoder contract: latent
        return self.encode(proteome)[0]  # mu (deterministic latent for transfer)

    @staticmethod
    def elbo(out: VAEOutput, target_epigenome: Tensor, beta: float = 1.0) -> Tensor:
        recon = nn.functional.mse_loss(out.recon, target_epigenome, reduction="mean")
        kl = -0.5 * torch.mean(1 + out.logvar - out.mu.pow(2) - out.logvar.exp())
        return recon + beta * kl


# --------------------------------------------------------------------------- #
# L3 — ESM-2 node features + STRING PPI GAT encoder
# --------------------------------------------------------------------------- #
class PPIGraphGAT(ModalityEncoder):
    """Graph-attention encoder over the STRING PPI graph with ESM-2 node features.

    ``torch_geometric`` is imported lazily so this file loads without it; the graph
    (edge_index, node features) is built by scripts/stage_d/build_string_graph.py and
    passed at construction. Produces a per-node embedding; a gene-set readout gives the
    modality latent consumed by fusion in Stage E."""

    def __init__(self, in_dim: int, latent_dim: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv  # noqa: F401
        except Exception as e:  # pragma: no cover - pre-install
            raise ImportError(
                "PPIGraphGAT needs torch_geometric (Stage-D interstage install)."
            ) from e
        from torch_geometric.nn import GATConv
        self.latent_dim = latent_dim
        dims = [in_dim] + [latent_dim] * layers
        self.convs = nn.ModuleList([
            GATConv(dims[i], dims[i + 1] // heads, heads=heads) for i in range(layers)
        ])

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        if edge_index is None:
            raise ValueError("PPIGraphGAT.forward needs edge_index (the STRING graph)")
        for conv in self.convs:
            x = torch.nn.functional.elu(conv(x, edge_index))
        return x  # (n_nodes, latent_dim) node embeddings


__all__ = ["CellLineVAE", "VAEOutput", "PPIGraphGAT"]
