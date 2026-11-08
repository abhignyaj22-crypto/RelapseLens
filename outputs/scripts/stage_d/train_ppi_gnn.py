#!/usr/bin/env python3
"""Stage D — PPI-GNN + degree-preserving-null gate (L3 exit criterion).

Task: predict DepMap CRISPR gene essentiality (mean Chronos gene-effect across cell
lines) from ESM-2 node features via message passing over the STRING PPI graph. The
honest gate is whether the REAL PPI graph beats a DEGREE-PRESERVING rewired null (and
an edgeless MLP) on held-out nodes — i.e. does PPI topology carry signal beyond node
features + degree? Real > null => the network prior is licensed for mechanism use.

Public data only (STRING v12, ESM-2, DepMap CRISPR). Reports held-out Spearman/R2 for
{real graph, degree-preserving null, no-edge MLP}; a null result is reportable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mm_tte_survival.models.foundation import PPIGraphGAT  # noqa: E402

DERIVED = ROOT / "data/external/_derived"
CRISPR = ROOT / "data/external/drug_response/CRISPRGeneEffect.csv"
OUT = ROOT / "outputs/stage_d"
SEED = 42


class NodeRegressor(nn.Module):
    """PPIGraphGAT node embeddings -> scalar readout (or MLP-only when use_graph=False)."""

    def __init__(self, in_dim, latent_dim=64, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        if use_graph:
            self.gnn = PPIGraphGAT(in_dim=in_dim, latent_dim=latent_dim, heads=4, layers=2)
            self.readout = nn.Linear(latent_dim, 1)
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ELU(),
                                     nn.Linear(latent_dim, latent_dim), nn.ELU())
            self.readout = nn.Linear(latent_dim, 1)

    def forward(self, x, edge_index):
        h = self.gnn(x, edge_index) if self.use_graph else self.mlp(x)
        return self.readout(h).squeeze(-1)


def degree_preserving_null(edge_index: torch.Tensor, n_nodes: int, seed: int) -> torch.Tensor:
    """Configuration-model rewire preserving each node's degree (networkx)."""
    import networkx as nx
    ei = edge_index.numpy()
    und = ei[:, ei[0] < ei[1]]  # one direction
    G = nx.Graph(); G.add_nodes_from(range(n_nodes)); G.add_edges_from(zip(und[0], und[1]))
    nx.double_edge_swap(G, nswap=min(5 * G.number_of_edges(), 2_000_000),
                        max_tries=20_000_000, seed=seed)
    u = np.array([e[0] for e in G.edges()]); v = np.array([e[1] for e in G.edges()])
    uu = np.concatenate([u, v]); vv = np.concatenate([v, u])
    return torch.tensor(np.stack([uu, vv]), dtype=torch.long)


def train_eval(x, edge_index, y, train_m, val_m, test_m, use_graph, dev, epochs=300):
    torch.manual_seed(SEED)
    model = NodeRegressor(x.shape[1], latent_dim=64, use_graph=use_graph).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(x, edge_index)
        loss = lossf(pred[train_m], y[train_m])
        loss.backward(); opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vp = model(x, edge_index)
                vl = float(lossf(vp[val_m], y[val_m]))
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
            if bad >= 12:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = model(x, edge_index)[test_m].cpu().numpy()
    t = y[test_m].cpu().numpy()
    return {"spearman": float(spearmanr(p, t).statistic),
            "r2": float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))}


def main() -> int:
    import pandas as pd
    rng = np.random.default_rng(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = np.load(DERIVED / "string_graph.npz", allow_pickle=True)
    nodes = list(g["node_names"]); node_idx = {gn: i for i, gn in enumerate(nodes)}
    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
    feat = np.load(DERIVED / "esm2_node_features.npz", allow_pickle=True)["features"]

    # CRISPR essentiality target (mean gene-effect across cell lines)
    cr = pd.read_csv(CRISPR, index_col=0)
    ess = cr.mean(axis=0)
    ess.index = [re.sub(r"\s*\(\d+\)$", "", c) for c in ess.index]
    y = np.full(len(nodes), np.nan, dtype="float32")
    for gene, val in ess.items():
        if gene in node_idx:
            y[node_idx[gene]] = val
    labeled = np.where(~np.isnan(y))[0]
    y = np.nan_to_num(y)

    # split FIRST, then standardize with TRAIN-only stats (leak-free): node-feature
    # and target mean/std are computed on TRAIN nodes only, then applied to all nodes.
    # (Previously standardized on the full set → leaked test-node stats into the gate.)
    perm = rng.permutation(labeled)
    n = len(perm); ntr, nva = int(0.7 * n), int(0.15 * n)
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]

    fmu, fsd = feat[tr].mean(0), feat[tr].std(0) + 1e-6
    x = (feat - fmu) / fsd
    ymu, ysd = y[tr].mean(), y[tr].std() + 1e-6
    yz = (y - ymu) / ysd

    def mask(ix):
        m = torch.zeros(len(nodes), dtype=torch.bool); m[ix] = True; return m.to(dev)

    X = torch.tensor(x, dtype=torch.float32).to(dev)
    Y = torch.tensor(yz, dtype=torch.float32).to(dev)
    ei = edge_index.to(dev)
    tr_m, va_m, te_m = mask(tr), mask(va), mask(te)

    print(f"[ppi_gnn] nodes={len(nodes)} labeled={len(labeled)} edges={ei.shape[1]//2} dev={dev}", flush=True)
    res = {}
    res["real_graph"] = train_eval(X, ei, Y, tr_m, va_m, te_m, True, dev)
    print("[ppi_gnn] real:", res["real_graph"], flush=True)
    null_ei = degree_preserving_null(edge_index, len(nodes), SEED).to(dev)
    res["degree_null"] = train_eval(X, null_ei, Y, tr_m, va_m, te_m, True, dev)
    print("[ppi_gnn] null:", res["degree_null"], flush=True)
    res["mlp_no_edges"] = train_eval(X, ei, Y, tr_m, va_m, te_m, False, dev)
    print("[ppi_gnn] mlp :", res["mlp_no_edges"], flush=True)

    res["gate_real_beats_null"] = bool(res["real_graph"]["spearman"] > res["degree_null"]["spearman"])
    res["gate_graph_beats_mlp"] = bool(res["real_graph"]["spearman"] > res["mlp_no_edges"]["spearman"])
    res["target"] = "DepMap CRISPR mean gene-effect (essentiality)"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ppi_gnn_null_gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
