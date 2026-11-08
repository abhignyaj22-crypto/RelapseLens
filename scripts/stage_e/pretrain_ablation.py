#!/usr/bin/env python3
"""Stage E — MEASURE the masked-gene pretrain's ΔC end-to-end (raw-gene encoder).

The proposed fusion model feeds the transcriptome encoder in-fold PCA(16). `training/
pretrain.masked_gene_pretrain` instead warm-starts a RAW-gene `TranscriptomeEncoder`
(input_dim = n_genes) by label-free masked reconstruction. This script wires that raw-gene
encoder into an end-to-end transcriptome→survival model and measures whether the pretrain
buys any discrimination, as a PAIRED ΔC (Uno IPCW) over repeated patient-disjoint splits:

    ΔC = C(raw genes, pretrained-init encoder) − C(raw genes, random-init encoder)

isolated in a transcriptome-only survival model (so the pretrain effect is not masked by the
other modalities, where omics is already a documented null). A pca16 arm is reported for
reference. Leak-free: genes standardized train-only per split; the pretrain is label-free and
fit on TRAIN patients only per split (no survival labels, no test rows); splits are patient-
disjoint. A ΔC CI that includes 0 is a reportable NULL.

Public MMRF-CoMMpass OPEN, OS endpoint. No synthetic data on the reported path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mm_tte_survival.models.encoders import TranscriptomeEncoder            # noqa: E402
from mm_tte_survival.models.survival_heads import CoxHead                   # noqa: E402
from mm_tte_survival.training.losses import cox_ph_loss                     # noqa: E402
from mm_tte_survival.training.pretrain import masked_gene_pretrain          # noqa: E402
from mm_tte_survival.metrics_ipcw import uno_c_index                        # noqa: E402
from mm_tte_survival.data.omics_pca import OmicsInFoldPCA                   # noqa: E402

REAL = ROOT / "data/real"
OUT = ROOT / "outputs/stage_e"
SEED = 42
LAT = 32
HIDDEN = 256


def fit_eval_transcriptome(Xz_tr, t_tr, e_tr, Xz_va, t_va, e_va, Xz_te, *,
                           input_dim, hidden, pretrained_state=None, seed=SEED,
                           epochs=300, lr=3e-3, device="cpu"):
    """Train TranscriptomeEncoder(input_dim)→CoxHead on standardized features and return the
    test risk vector. `pretrained_state` (optional) warm-starts the encoder; otherwise random
    init. Early-stops on validation Cox NLL. Deterministic given `seed`."""
    torch.manual_seed(seed)
    enc = TranscriptomeEncoder(input_dim, latent_dim=LAT, hidden=hidden, dropout=0.1).to(device)
    if pretrained_state is not None:
        enc.load_state_dict(pretrained_state)
    head = CoxHead(LAT).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    Xtr = torch.tensor(Xz_tr, dtype=torch.float32, device=device)
    ttr = torch.tensor(t_tr, dtype=torch.float32, device=device)
    etr = torch.tensor(e_tr, dtype=torch.float32, device=device)
    Xva = torch.tensor(Xz_va, dtype=torch.float32, device=device)
    tva = torch.tensor(t_va, dtype=torch.float32, device=device)
    eva = torch.tensor(e_va, dtype=torch.float32, device=device)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        enc.train(); head.train(); opt.zero_grad()
        risk = head(enc(Xtr)).squeeze(-1)
        loss = cox_ph_loss(risk, ttr, etr, ties="efron")
        loss.backward(); opt.step()
        if ep % 5 == 0:
            enc.eval(); head.eval()
            with torch.no_grad():
                vl = float(cox_ph_loss(head(enc(Xva)).squeeze(-1), tva, eva, ties="efron"))
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {"e": {k: v.cpu().clone() for k, v in enc.state_dict().items()},
                              "h": {k: v.cpu().clone() for k, v in head.state_dict().items()}}
            else:
                bad += 1
            if bad >= 10:
                break
    if best_state:
        enc.load_state_dict(best_state["e"]); head.load_state_dict(best_state["h"])
    enc.eval(); head.eval()
    with torch.no_grad():
        Xte = torch.tensor(Xz_te, dtype=torch.float32, device=device)
        return head(enc(Xte)).squeeze(-1).cpu().numpy()


def _standardize(train_rows, all_rows):
    mu = train_rows.mean(0); sd = train_rows.std(0) + 1e-6
    return (all_rows - mu) / sd


def run_ablation(gene_df: pd.DataFrame, surv: pd.DataFrame, *, splits=10, seed=SEED,
                 pretrain_epochs=300, n_hvg=2000, with_pca_arm=True, device=None) -> dict:
    """Paired ΔC ablation (pretrained vs random raw-gene encoder) over patient-disjoint splits.

    gene_df: (n_patients × n_genes) raw expression, index=patient_id.
    surv:    columns patient_id, time_months, event (aligned subset).
    n_hvg:   restrict the raw-gene encoder to the top-`n_hvg` TRAIN-variance genes per split
             (leak-free HVG selection). The full 19.5k-gene matrix is intractable for a masked
             autoencoder at this N (n≪p); HVG keeps it a genuine raw-gene encoder while letting
             the pretrain actually learn (masked_r2 > 0). Pass 0 to disable.
    Returns the aggregated result dict (means, CIs, per-arm C, sanity masked_r2)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ids = [p for p in surv["patient_id"].astype(str) if p in gene_df.index]
    surv = surv[surv["patient_id"].astype(str).isin(ids)].reset_index(drop=True)
    G = gene_df.loc[surv["patient_id"].astype(str)].values.astype("float32")   # rows aligned to surv
    G = G[:, G.var(0) > 0]
    t = surv["time_months"].clip(lower=0.1).values.astype(float)
    e = surv["event"].astype(int).values
    n, g_all = G.shape

    dC_pre_rand, C_pre, C_rand, C_pca, R2 = [], [], [], [], []
    g_used = g_all
    for r in range(splits):
        rs = np.random.default_rng(seed + r)
        ix = rs.permutation(n); cut = int(0.7 * n); vcut = int(0.85 * n)
        tr, va, te = ix[:cut], ix[cut:vcut], ix[vcut:]
        if e[te].sum() < 8 or e[tr].sum() < 5:
            continue
        # leak-free HVG selection: top-variance genes on TRAIN rows only
        if n_hvg and g_all > n_hvg:
            hvg = np.argsort(G[tr].var(0))[::-1][:n_hvg]
            Graw = G[:, hvg]
        else:
            Graw = G
        g_used = Graw.shape[1]
        Xz = _standardize(Graw[tr], Graw)                  # train-only standardization
        # label-free pretrain on TRAIN patients only (no survival, no test rows)
        pre = masked_gene_pretrain(Graw[tr], latent_dim=LAT, hidden=HIDDEN,
                                   epochs=pretrain_epochs, seed=seed, device=dev)
        R2.append(pre.masked_r2)
        pre_state = {k: v.to(dev) for k, v in pre.encoder.state_dict().items()}
        common = dict(input_dim=g_used, hidden=HIDDEN, seed=seed + r, device=dev)
        risk_pre = fit_eval_transcriptome(Xz[tr], t[tr], e[tr], Xz[va], t[va], e[va], Xz[te],
                                          pretrained_state=pre_state, **common)
        risk_rand = fit_eval_transcriptome(Xz[tr], t[tr], e[tr], Xz[va], t[va], e[va], Xz[te],
                                           pretrained_state=None, **common)
        c_pre = uno_c_index(t[tr], e[tr], t[te], e[te], risk_pre)
        c_rand = uno_c_index(t[tr], e[tr], t[te], e[te], risk_rand)
        C_pre.append(c_pre); C_rand.append(c_rand); dC_pre_rand.append(c_pre - c_rand)
        if with_pca_arm:
            gdf = pd.DataFrame(G, index=surv["patient_id"].astype(str).values)
            pca = OmicsInFoldPCA(k=16).fit(gdf, gdf.index[tr].tolist())
            P = pca.transform(gdf).values.astype("float32")
            Pz = _standardize(P[tr], P)
            risk_pca = fit_eval_transcriptome(Pz[tr], t[tr], e[tr], Pz[va], t[va], e[va], Pz[te],
                                              input_dim=P.shape[1], hidden=64, seed=seed + r, device=dev)
            C_pca.append(uno_c_index(t[tr], e[tr], t[te], e[te], risk_pca))

    dC = np.array(dC_pre_rand, float)
    def ci(a):
        a = np.asarray(a, float)
        return None if a.size == 0 else [float(np.nanmean(a)),
                                         float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {
        "measurement": "paired ΔC(Uno IPCW): raw-gene pretrained-init encoder − random-init encoder",
        "endpoint": "overall_survival (public MMRF-CoMMpass OPEN)",
        "model": f"TranscriptomeEncoder(n_genes={g_used}, latent={LAT}, hidden={HIDDEN}) → CoxHead (Efron)",
        "n_patients": int(n), "n_genes_total": int(g_all), "n_genes_used_hvg": int(g_used),
        "n_splits": int(dC.size), "device": dev,
        "pretrain": "masked_gene_pretrain, label-free, TRAIN-only per split (top-variance HVG)",
        "masked_r2_mean": float(np.nanmean(R2)) if R2 else float("nan"),
        "C_pretrained_mean": float(np.nanmean(C_pre)) if C_pre else float("nan"),
        "C_random_mean": float(np.nanmean(C_rand)) if C_rand else float("nan"),
        "C_pca16_mean": float(np.nanmean(C_pca)) if C_pca else None,
        "dC_pretrained_minus_random_mean_ci": ci(dC),
        "gate_pretrain_helps": bool(dC.size and np.nanpercentile(dC, 2.5) > 0),
        "note": ("dC CI excludes 0 ⇒ masked-gene pretraining yields a better survival-predictive "
                 "transcriptome encoder; else pre-registered NULL. masked_r2>0 confirms the pretrain "
                 "learned real gene-gene structure regardless of the downstream survival ΔC."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="fast: 3 splits, shorter pretrain")
    args = ap.parse_args()

    d = np.load(REAL / "gene_matrix.npz", allow_pickle=True)
    gene_df = pd.DataFrame(d["matrix"], index=[str(p) for p in d["patient_ids"]], columns=list(d["genes"]))
    clin = pd.read_csv(REAL / "clinical_survival.csv")
    clin["patient_id"] = clin["patient_id"].astype(str)
    surv = clin[["patient_id", "time_months", "event"]].dropna()

    res = run_ablation(gene_df, surv,
                       splits=3 if args.smoke else args.splits,
                       pretrain_epochs=60 if args.smoke else 150)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pretrain_ablation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
