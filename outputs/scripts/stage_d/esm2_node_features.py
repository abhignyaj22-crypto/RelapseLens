#!/usr/bin/env python3
"""Stage D — ESM-2 protein embeddings as PPI-graph node features (L3).

Real protein sequences (STRING v12 FASTA) -> ESM-2 mean-pooled per-protein embeddings,
aligned to the string_graph node order. Computed once on GPU and cached; these are the
node features for PPIGraphGAT. Multiple ENSP mapping to one gene -> keep the longest
sequence (canonical isoform proxy).

Output data/external/_derived/esm2_node_features.npz:
    node_names (n,)  object
    features   (n,d) float32   (d = ESM-2 hidden size; 0-vector if no sequence found)
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PPI = ROOT / "data/external/ppi"
SEQS = PPI / "9606.protein.sequences.v12.0.fa.gz"
INFO = PPI / "9606.protein.info.v12.0.txt.gz"
GRAPH = ROOT / "data/external/_derived/string_graph.npz"
OUT = ROOT / "data/external/_derived/esm2_node_features.npz"

MODEL = "facebook/esm2_t12_35M_UR50D"
MAX_LEN = 1022
BATCH = 32


def parse_fasta(path: Path) -> dict[str, str]:
    seqs, cur, buf = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs[cur] = "".join(buf)
                cur = line[1:].split()[0].strip()
                buf = []
            else:
                buf.append(line.strip())
    if cur:
        seqs[cur] = "".join(buf)
    return seqs


def ensp_to_gene() -> dict[str, str]:
    import pandas as pd
    with gzip.open(INFO, "rt") as fh:
        info = pd.read_csv(fh, sep="\t")
    col = info.columns[0]
    return dict(zip(info[col], info["preferred_name"]))


def main() -> int:
    from transformers import AutoTokenizer, AutoModel

    nodes = list(np.load(GRAPH, allow_pickle=True)["node_names"])
    node_idx = {g: i for i, g in enumerate(nodes)}
    ensp_seq = parse_fasta(SEQS)
    e2g = ensp_to_gene()

    # gene -> longest sequence among its ENSPs
    gene_seq: dict[str, str] = {}
    for ensp, seq in ensp_seq.items():
        g = e2g.get(ensp)
        if g in node_idx and len(seq) > len(gene_seq.get(g, "")):
            gene_seq[g] = seq[:MAX_LEN]

    covered = [g for g in nodes if g in gene_seq]
    print(f"[esm2] nodes={len(nodes)} with_sequence={len(covered)} "
          f"({100*len(covered)/len(nodes):.1f}%)", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    d = model.config.hidden_size
    feats = np.zeros((len(nodes), d), dtype="float32")

    with torch.no_grad():
        for s in range(0, len(covered), BATCH):
            batch_genes = covered[s:s + BATCH]
            enc = tok([gene_seq[g] for g in batch_genes], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN).to(dev)
            out = model(**enc).last_hidden_state           # (B, L, d)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            for g, vec in zip(batch_genes, pooled.cpu().numpy()):
                feats[node_idx[g]] = vec
            if (s // BATCH) % 20 == 0:
                print(f"[esm2] {s+len(batch_genes)}/{len(covered)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, node_names=np.array(nodes, dtype=object), features=feats)
    print(f"[esm2] wrote {OUT.relative_to(ROOT)} dim={d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
