#!/usr/bin/env python3
"""Part II Stages 4-5 — train the dynamic flow→FHT model and run the DECISIVE gate.

Architecture (honest, tractable): the disease latent is the transcriptome encoder output
(in-fold PCA → MLP). A population velocity FIELD (FlowMatchingDynamics, unconditional) evolves
it along disease latent-time; the CFM loss shapes that field from the PUBLIC stage marginals
(GSE271107 ρ_HD→ρ_MGUS→ρ_SMM→ρ_MM). The event is the latent's first passage across a barrier:
the FHT reads the CALENDAR clock from the CoMMpass OS labels (two-part identification — field
from atlases, rate from survival). Patient-specificity enters through each patient's initial
state z0 and the barrier, so even an unconditional field yields a patient-specific drift
ν = wᵀ·v̄(z0).

THE GATE (Stage 5, pre-registered): the trained flow must beat BOTH (a) StaticDynamics
(drift≡0; risk from initial readout only) AND (b) a random/frozen field (untrained
FlowMatchingDynamics) on paired ΔC (Uno IPCW), with a percentile CI excluding 0. Otherwise we
report the NULL and keep static — a legitimate outcome. Public-only; in-fold PCA; no synthetic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from mm_tte_survival.data.omics_pca import OmicsInFoldPCA                       # noqa: E402
from mm_tte_survival.data.stage_marginals import STAGES, load_stage_marginals   # noqa: E402
from mm_tte_survival.longitudinal.crossing_head import crossing_risk_score      # noqa: E402
from mm_tte_survival.models.dynamics import (FlowMatchingDynamics,              # noqa: E402
                                             StaticDynamics)
from mm_tte_survival.models.encoders import TranscriptomeEncoder                # noqa: E402
from mm_tte_survival.training.flow_losses import (                             # noqa: E402
    conditional_flow_matching_loss, dynamic_fht_nll)

REAL = ROOT / "data/real"
OUT = ROOT / "outputs/flow"
SEED, R, EPOCHS, LAT, NPC = 42, 12, 150, 32, 16
LAMBDA_CFM = 0.5


def _sp(x):  # softplus
    return torch.nn.functional.softplus(x)


class FlowSurv(torch.nn.Module):
    """transcriptome encoder + monotone readout + barrier/sigma + latent-time dynamics."""

    def __init__(self, dynamics):
        super().__init__()
        self.enc = TranscriptomeEncoder(NPC, latent_dim=LAT, hidden=64, dropout=0.1)
        self.dynamics = dynamics
        self.raw_w = torch.nn.Parameter(torch.zeros(LAT))       # w = softplus(raw_w) >= 0 (monotone)
        self.raw_a = torch.nn.Parameter(torch.tensor(1.0))      # barrier a = softplus(raw_a) > 0
        self.raw_s = torch.nn.Parameter(torch.zeros(1))         # sigma = softplus(raw_s) > 0

    def z0(self, pcs):
        return self.enc(pcs)

    def _params(self, n, dev):
        # sigma floored away from 0 for first-passage numerical stability
        return _sp(self.raw_w), _sp(self.raw_a).expand(n), (_sp(self.raw_s) + 0.1).expand(n)

    def fht_nll(self, pcs, time, event):
        n = pcs.shape[0]
        w, a, sig = self._params(n, pcs.device)
        c = torch.zeros(n, 0, device=pcs.device)
        return dynamic_fht_nll(self.dynamics, self.z0(pcs), c, w, a,
                               torch.ones(n, device=pcs.device), time, event, sig)

    @torch.no_grad()
    def risk(self, pcs, horizon):
        n = pcs.shape[0]
        w, a, sig = self._params(n, pcs.device)
        c = torch.zeros(n, 0, device=pcs.device)
        z0 = self.z0(pcs)
        _, v_mean = self.dynamics.flow(z0, c, torch.ones(n, device=pcs.device))
        drift = (v_mean * w).sum(-1).clamp(-20.0, 20.0)      # guard the untrained random field
        gap = (a - (z0 * w).sum(-1)).clamp_min(1e-6)
        risk = crossing_risk_score(gap, drift, sig, float(horizon)).cpu().numpy()
        return np.nan_to_num(risk, nan=0.5, posinf=1.0, neginf=0.0)


def uno_c(tr_df, te_df, risk):
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
    ytr = Surv.from_arrays(tr_df["event"].astype(bool), tr_df["time_months"].clip(lower=0.1))
    yte = Surv.from_arrays(te_df["event"].astype(bool), te_df["time_months"].clip(lower=0.1))
    return float(concordance_index_ipcw(ytr, yte, risk)[0])


def _make(kind, dev, seed):
    torch.manual_seed(seed)
    if kind == "static":
        dyn = StaticDynamics(LAT)
    else:
        dyn = FlowMatchingDynamics(LAT, cond_dim=0, hidden=128, n_steps=8)
        if kind == "random":                      # frozen untrained field (ablation)
            for p in dyn.parameters():
                p.requires_grad_(False)
    return FlowSurv(dyn).to(dev)


def _stage_latents(model, stage_pcs, dev):
    """Embed the (already PCA-projected) stage clouds with the LIVE encoder → latent clouds."""
    return {s: model.z0(torch.tensor(stage_pcs[s], dtype=torch.float32, device=dev))
            for s in STAGES}


def train_variant(kind, tr_pcs, tr_t, tr_e, stage_pcs, dev, seed):
    model = _make(kind, dev, seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-3)
    use_cfm = kind == "dynamic"
    pairs = list(zip(STAGES[:-1], STAGES[1:]))
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        loss = model.fht_nll(tr_pcs, tr_t, tr_e).mean()
        if use_cfm:
            zc = _stage_latents(model, stage_pcs, dev)
            cfm = 0.0
            for a, b in pairs:
                m = min(zc[a].shape[0], zc[b].shape[0], 128)
                ia = torch.randperm(zc[a].shape[0], device=dev)[:m]
                ib = torch.randperm(zc[b].shape[0], device=dev)[:m]
                cfm = cfm + conditional_flow_matching_loss(
                    model.dynamics, zc[a][ia], zc[b][ib], torch.zeros(m, 0, device=dev), sigma=0.1)
            loss = loss + LAMBDA_CFM * cfm / len(pairs)
        if not torch.isfinite(loss):
            continue                                   # skip a rare unstable step (untrained field)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    return model


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)

    clin = pd.read_csv(REAL / "clinical_survival.csv"); clin["patient_id"] = clin["patient_id"].astype(str)
    d = np.load(REAL / "gene_matrix.npz", allow_pickle=True)
    gene_df = pd.DataFrame(d["matrix"], index=[str(p) for p in d["patient_ids"]], columns=list(d["genes"]))
    gene_df = gene_df.loc[:, gene_df.var(axis=0) > 0]
    df = clin[["patient_id", "time_months", "event"]].dropna()
    df = df[df["patient_id"].isin(gene_df.index)].reset_index(drop=True)

    try:
        stage_expr = load_stage_marginals()
    except FileNotFoundError:
        print("stage marginals not built; run mm_tte_survival.data.stage_marginals first")
        return 1
    commpass_genes = list(gene_df.columns)
    # stage clouds are cells × 19938 CoMMpass-ordered genes (same universe as gene_df, pre-filter)
    stage_full = {s: pd.DataFrame(stage_expr[s],
                  columns=list(np.load(ROOT / "data/external/scrna/stage_marginals/genes.npy",
                                       allow_pickle=True))) for s in STAGES}

    ev = df["event"].astype(int).values
    variants = ["dynamic", "static", "random"]
    C = {k: [] for k in variants}
    for r in range(R):
        rs = np.random.default_rng(SEED + r)
        i0, i1 = np.where(ev == 0)[0], np.where(ev == 1)[0]
        rs.shuffle(i0); rs.shuffle(i1)
        c0, c1 = int(0.7 * len(i0)), int(0.7 * len(i1))
        tr_i = np.concatenate([i0[:c0], i1[:c1]]); te_i = np.concatenate([i0[c0:], i1[c1:]])
        tr_df, te_df = df.iloc[tr_i], df.iloc[te_i]
        if te_df["event"].sum() < 10:
            continue
        # in-fold PCA fit on TRAIN patients; project patients AND stage clouds through it
        pca = OmicsInFoldPCA(k=NPC).fit(gene_df, tr_df["patient_id"].tolist())
        allpc = pca.transform(gene_df.loc[df["patient_id"].values]).values.astype("float32")
        # standardize PCs to the TRAIN-patient distribution (leak-free) so the encoder sees
        # comparable, bounded inputs; the scRNA stage cells are off-distribution (domain shift)
        # so they get the same transform then clipped to keep the CFM cost matrix finite.
        pc_mu, pc_sd = allpc[tr_i].mean(0), allpc[tr_i].std(0) + 1e-6
        allpc = (allpc - pc_mu) / pc_sd
        stage_pcs = {s: np.clip((pca.transform(stage_full[s][pca.top_genes_]).values - pc_mu) / pc_sd,
                                -8.0, 8.0).astype("float32") for s in STAGES}
        tr_pcs = torch.tensor(allpc[tr_i], device=dev)
        te_pcs = torch.tensor(allpc[te_i], device=dev)
        tr_t = torch.tensor(tr_df["time_months"].clip(lower=0.1).values, dtype=torch.float32, device=dev)
        tr_e = torch.tensor(tr_df["event"].astype(float).values, dtype=torch.float32, device=dev)
        horizon = float(np.median(tr_df["time_months"].clip(lower=0.1)))
        for k in variants:
            m = train_variant(k, tr_pcs, tr_t, tr_e, stage_pcs, dev, SEED + r)
            C[k].append(uno_c(tr_df, te_df, m.risk(te_pcs, horizon)))

    def ci(a):
        a = np.array(a); return [float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    dyn, sta, ran = np.array(C["dynamic"]), np.array(C["static"]), np.array(C["random"])
    dC_static, dC_random = dyn - sta, dyn - ran
    beats_static = float(np.percentile(dC_static, 2.5)) > 0
    beats_random = float(np.percentile(dC_random, 2.5)) > 0
    res = {
        "n_splits": len(dyn),
        "C_dynamic": ci(dyn), "C_static": ci(sta), "C_random_field": ci(ran),
        "dC_dynamic_vs_static": ci(dC_static),
        "dC_dynamic_vs_random_field": ci(dC_random),
        "gate_flow_beats_static": beats_static,
        "gate_flow_beats_random_field": beats_random,
        "PROMOTE_dynamic": bool(beats_static and beats_random),
        "verdict": ("PROMOTE: flow adds discrimination beyond static AND a random field"
                    if (beats_static and beats_random) else
                    "NULL: flow does not beat static and/or the random-field ablation — keep static"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flow_gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
