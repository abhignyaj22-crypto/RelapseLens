#!/usr/bin/env python3
"""Stage D — build the STRING v12 PPI graph over the patient gene space (L3).

Real public network data (STRING v12, CC BY 4.0). Maps ENSP -> gene symbol via the
STRING info file (preferred_name), restricts nodes to the patient transcriptome gene
space, keeps high-confidence edges (combined_score >= threshold), and emits an
undirected edge_index for the PPI-GNN.

Output data/external/_derived/string_graph.npz:
    node_names (n,)   object   gene symbols (aligned to node index)
    edge_index (2, E) int64    undirected edges (both directions)
    ensp2gene  dict via node_names; combined_score threshold in meta
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PPI = ROOT / "data/external/ppi"
LINKS = PPI / "9606.protein.links.full.v12.0.txt.gz"
INFO = PPI / "9606.protein.info.v12.0.txt.gz"
GENES = ROOT / "data/real/gene_matrix.npz"
OUT = ROOT / "data/external/_derived/string_graph.npz"

SCORE_THRESHOLD = 700  # STRING "high confidence"


def ensp_to_gene() -> dict[str, str]:
    with gzip.open(INFO, "rt") as fh:
        info = pd.read_csv(fh, sep="\t")
    col = "#string_protein_id" if "#string_protein_id" in info.columns else info.columns[0]
    return dict(zip(info[col], info["preferred_name"]))


def main() -> int:
    patient_genes = set(np.load(GENES, allow_pickle=True)["genes"].tolist())
    e2g = ensp_to_gene()

    # stream links, keep high-confidence edges among patient genes
    usecols = ["protein1", "protein2", "combined_score"]
    edges_g = []
    with gzip.open(LINKS, "rt") as fh:
        for chunk in pd.read_csv(fh, sep=" ", usecols=usecols, chunksize=2_000_000):
            chunk = chunk[chunk["combined_score"] >= SCORE_THRESHOLD]
            g1 = chunk["protein1"].map(e2g)
            g2 = chunk["protein2"].map(e2g)
            keep = g1.notna() & g2.notna() & g1.isin(patient_genes) & g2.isin(patient_genes) & (g1 != g2)
            edges_g.append(pd.DataFrame({"g1": g1[keep].values, "g2": g2[keep].values}))
    edf = pd.concat(edges_g, ignore_index=True)

    # node universe = genes participating in >=1 kept edge
    nodes = sorted(set(edf["g1"]) | set(edf["g2"]))
    idx = {g: i for i, g in enumerate(nodes)}
    src = edf["g1"].map(idx).to_numpy()
    dst = edf["g2"].map(idx).to_numpy()
    # undirected: add both directions, dedup
    u = np.concatenate([src, dst]); v = np.concatenate([dst, src])
    pair = np.unique(np.stack([u, v], axis=1), axis=0)
    edge_index = pair.T.astype("int64")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, node_names=np.array(nodes, dtype=object),
                        edge_index=edge_index, threshold=SCORE_THRESHOLD)
    deg = np.bincount(edge_index[0], minlength=len(nodes))
    print(f"[string_graph] nodes={len(nodes)} edges(undirected)={edge_index.shape[1]//2} "
          f"avg_degree={deg.mean():.1f} max_degree={deg.max()} threshold={SCORE_THRESHOLD}")
    print(f"[string_graph] wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
