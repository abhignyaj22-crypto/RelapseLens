#!/usr/bin/env python3
"""Stage D — train the CellLineVAE (L1) on real paired CCLE proteome→epigenome.

Honest evaluation: held-out per-mark Spearman of predicted vs true histone-PTM,
against a predict-train-mean baseline (Spearman ~ 0). This tests whether the cell-line
proteome carries recoverable epigenome signal — the licence for the L1 representation.
Small n (367) so we regularize hard (low latent dim, dropout, weight decay, early stop)
and report val honestly (a null here is a legitimate, reportable outcome).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mm_tte_survival.models.foundation import CellLineVAE  # noqa: E402

PAIRED = ROOT / "data/external/_derived/ccle_paired.npz"
OUT = ROOT / "outputs/stage_d"
SEED = 42


def main() -> int:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(PAIRED, allow_pickle=True)
    X = torch.tensor(d["proteome"], dtype=torch.float32)
    Y = torch.tensor(d["epigenome"], dtype=torch.float32)
    n, p = X.shape
    marks = list(d["marks"])

    idx = rng.permutation(n)
    n_val = max(40, n // 5)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    # standardize with TRAIN-only stats (prep now emits raw matrices — leak-free)
    xmu, xsd = X[tr_i].mean(0), X[tr_i].std(0).clamp_min(1e-6)
    ymu, ysd = Y[tr_i].mean(0), Y[tr_i].std(0).clamp_min(1e-6)
    X = (X - xmu) / xsd
    Y = (Y - ymu) / ysd
    Xtr, Ytr = X[tr_i].to(dev), Y[tr_i].to(dev)
    Xva, Yva = X[val_i].to(dev), Y[val_i].to(dev)

    model = CellLineVAE(proteome_dim=p, epigenome_dim=Y.shape[1],
                        latent_dim=32, hidden=256, dropout=0.3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_val, best_state, patience, bad = 1e9, None, 40, 0
    for epoch in range(600):
        model.train()
        opt.zero_grad()
        out = model.vae_forward(Xtr)
        loss = model.elbo(out, Ytr, beta=0.5)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == 599:
            model.eval()
            with torch.no_grad():
                vout = model.vae_forward(Xva)
                vloss = float(model.elbo(vout, Yva, beta=0.5))
            if vloss < best_val - 1e-4:
                best_val, best_state, bad = vloss, {k: v.detach().cpu().clone()
                                                    for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
            if bad * 10 >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model.vae_forward(Xva).recon.cpu().numpy()
    true = Yva.cpu().numpy()
    train_mean = Ytr.mean(dim=0).cpu().numpy()

    # per-mark held-out Spearman: VAE vs a REAL permutation null (shuffle predictions
    # vs true). This computes the noise floor honestly instead of asserting null=0 —
    # important because with small n_val the Spearman noise floor is ~1/sqrt(n_val).
    K_PERM = 200
    per_mark, null_means = [], []
    for j, mk in enumerate(marks):
        if np.std(true[:, j]) < 1e-8:
            continue
        rho_vae = spearmanr(pred[:, j], true[:, j]).statistic
        null = np.array([spearmanr(rng.permutation(pred[:, j]), true[:, j]).statistic
                         for _ in range(K_PERM)])
        null_p95 = float(np.percentile(null, 95))
        per_mark.append({"mark": mk, "spearman_vae": float(rho_vae),
                         "null_p95": null_p95, "beats_perm_null": bool(rho_vae > null_p95)})
        null_means.append(float(np.mean(null)))
    rhos = np.array([m["spearman_vae"] for m in per_mark])
    beats = np.array([m["beats_perm_null"] for m in per_mark])

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "n_train": len(tr_i), "n_val": len(val_i), "n_proteins": p,
        "n_marks_eval": len(per_mark), "best_val_elbo": best_val,
        "median_spearman": float(np.median(rhos)), "mean_spearman": float(np.mean(rhos)),
        "frac_marks_positive": float(np.mean(rhos > 0)),
        "frac_marks_spearman_gt_0.2": float(np.mean(rhos > 0.2)),
        "permutation_null_mean_spearman": float(np.mean(null_means)),
        "spearman_noise_floor_approx": float(1.0 / np.sqrt(max(len(val_i), 2))),
        "frac_marks_beat_perm_null_p95": float(np.mean(beats)),
        "baseline": f"permutation null (K={K_PERM} shuffles/mark); a mark 'counts' only if it beats its own null p95",
        "per_mark": sorted(per_mark, key=lambda m: -m["spearman_vae"])[:10],
        "device": dev,
    }
    (OUT / "ccle_vae_recon.json").write_text(json.dumps(result, indent=2))
    torch.save(best_state, OUT / "ccle_vae_encoder.pt")
    # save the cell-line latent (for Stage E transfer)
    with torch.no_grad():
        z_all = model(X.to(dev)).cpu().numpy()
    np.savez_compressed(OUT / "ccle_latent.npz", latent=z_all.astype("float32"),
                        cell_lines=d["cell_lines"])
    print(json.dumps({k: result[k] for k in
                      ["n_train", "n_val", "median_spearman", "mean_spearman",
                       "permutation_null_mean_spearman", "spearman_noise_floor_approx",
                       "frac_marks_beat_perm_null_p95", "device"]}, indent=2))
    print("top marks:", [(m["mark"], round(m["spearman_vae"], 2)) for m in result["per_mark"][:5]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
