#!/usr/bin/env python3
"""Stage E — the proposed multi-modal patient model (ResistanceModel end-to-end).

This is the dashboard proposal made concrete on real public data: per-modality encoders
-> CrossAttentionFusion (with modality dropout) -> Cox survival head, i.e.
ResistanceModel({clinical, transcriptome, programs, transferred_cellline_latent},
CrossAttentionFusion, CoxHead), trained on CoMMpass OS.

Evaluation goes through the ONE shared rigor stack (no hand-rolled copies):
  * discrimination  — metrics_ipcw.uno_c_index (Uno IPCW) + cumulative_dynamic_auc_mean
  * calibration     — survival_curves.ipcw_ibs + metrics_ipcw.d_calibration (via a
                      train-only Breslow baseline for the fusion Cox head)
  * baselines       — comparators.build_comparators: the fusion ΔC is benchmarked not only
                      against a clinical-only Cox but against the real ML zoo
                      (PenalizedCox / RSF / Coxnet / GradientBoosting / … auto-detected)
  * claim gating    — endpoints.gate_claims: an OS run can license only a technical-
                      validation claim, never a relapse/PFS/biological one (pre-registered)

All over repeated patient-disjoint splits; a null (ΔC CI includes 0) is a reportable result.

Patient modalities (all real, public MMRF-CoMMpass OPEN + the Stage-E transferred latent):
  clinical (7) · transcriptome PCA (16, in-fold) · mmSYGNAL programs (10) · transferred
  cell-line multi-omic latent (32, transcriptome-mediated from CCLE).
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
from mm_tte_survival.models.encoders import TranscriptomeEncoder as MLPEnc  # noqa: E402  (generic MLP modality encoder)
from mm_tte_survival.models.fusion import CrossAttentionFusion           # noqa: E402
from mm_tte_survival.models.survival_heads import CoxHead                # noqa: E402
from mm_tte_survival.models.resistance_model import ResistanceModel      # noqa: E402
from mm_tte_survival.data.omics_pca import OmicsInFoldPCA                 # noqa: E402  (leak-free in-fold PCA)
from mm_tte_survival.metrics_ipcw import (uno_c_index,                   # noqa: E402
                                          cumulative_dynamic_auc_mean, d_calibration)
from mm_tte_survival.survival_curves import (time_grid, breslow_baseline,  # noqa: E402
                                             cox_survival, ipcw_ibs)
from mm_tte_survival.comparators import build_comparators, backend_report, PenalizedCox  # noqa: E402
from mm_tte_survival.endpoints import gate_claims                        # noqa: E402

REAL = ROOT / "data/real"
OUT = ROOT / "outputs/stage_e"
SEED = 42
CLIN = ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy", "albumin", "b2m"]
PROG = ["prog_proliferation", "prog_mmset_t4_14", "prog_ccnd1_t11_14", "prog_maf_program",
        "prog_myc_targets", "prog_nfkb", "prog_ifn_response", "prog_plasma_cell_diff",
        "prog_bone_marrow_stroma", "prog_highrisk_gep_proxy"]
LAT = 32
N_PC = 16  # transcriptome PCs, fit IN-FOLD (never the leaky full-cohort omics.csv)


def zfill(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.fillna(df.median()).fillna(0.0)


def load_gene_df():
    """Raw patient bulk-RNA substrate (log2 TPM+1), index=patient_id. Drops the 428
    constant all-zero genes flagged in the data-quality sweep. PCA is fit train-only
    per split via OmicsInFoldPCA — omics.csv (full-cohort PCs) is NOT used here."""
    d = np.load(REAL / "gene_matrix.npz", allow_pickle=True)
    g = pd.DataFrame(d["matrix"], index=[str(p) for p in d["patient_ids"]], columns=list(d["genes"]))
    return g.loc[:, g.var(axis=0) > 0]


def load_patient_blocks():
    clin = pd.read_csv(REAL / "clinical_survival.csv"); clin["patient_id"] = clin["patient_id"].astype(str)
    prog = pd.read_csv(REAL / "program_activity.csv"); prog["patient_id"] = prog["patient_id"].astype(str)
    tz = pd.read_csv(OUT / "patient_transferred_latent.csv", index_col=0); tz.index = tz.index.astype(str)
    gene_df = load_gene_df()
    df = clin[["patient_id", "time_months", "event"] + CLIN].merge(
        prog[["patient_id"] + PROG], on="patient_id").merge(
        tz, left_on="patient_id", right_index=True)
    df = df[df["patient_id"].isin(gene_df.index)]
    df = df[df["time_months"].notna() & df["event"].notna()].reset_index(drop=True)
    # transcriptome block is materialised in-fold; declare its column names up front
    tpcs = [f"PC{i+1}" for i in range(N_PC)]
    blocks = {"clinical": CLIN, "transcriptome": tpcs, "programs": PROG, "transferred": list(tz.columns)}
    return df, blocks, gene_df


def make_model(blocks, dev):
    torch.manual_seed(SEED)
    encoders = {m: MLPEnc(len(cols), latent_dim=LAT, hidden=64, dropout=0.2)
                for m, cols in blocks.items()}
    fusion = CrossAttentionFusion({m: LAT for m in blocks}, d_model=LAT, n_heads=4,
                                  modality_dropout=0.2, out_dim=LAT)
    return ResistanceModel(encoders, fusion, CoxHead(LAT)).to(dev)


def to_batch(df, blocks, idx, scaler, dev):
    b = {}
    for m, cols in blocks.items():
        X = zfill(df.loc[idx, cols]).values.astype("float32")
        mu, sd = scaler[m]
        b[m] = torch.tensor(((X - mu) / sd).astype("float32"), device=dev)
    return b


def _arr(dfrow, col):
    return pd.to_numeric(dfrow[col]).clip(lower=0.1).values.astype(float)


def _uno(train_df, test_df, risk_test):
    """Uno IPCW concordance via the shared metric (higher risk = earlier event)."""
    return uno_c_index(_arr(train_df, "time_months"), train_df["event"].astype(int).values,
                       _arr(test_df, "time_months"), test_df["event"].astype(int).values,
                       np.asarray(risk_test, float))


def infold_transcriptome(df, gene_df, train_pos):
    """Fit PCA train-only on the raw gene matrix, transform ALL rows → a fold-local
    df with PC1..PCk columns. This is the leak-free replacement for omics.csv."""
    train_ids = df.loc[train_pos, "patient_id"].tolist()
    pca = OmicsInFoldPCA(k=N_PC).fit(gene_df, train_ids)
    pc = pca.transform(gene_df.loc[df["patient_id"].values])  # rows in df order
    fdf = df.copy()
    for i in range(pca.n_components_):
        fdf[f"PC{i+1}"] = pc.iloc[:, i].values
    for i in range(pca.n_components_, N_PC):  # pad if a fold yields <N_PC comps
        fdf[f"PC{i+1}"] = 0.0
    return fdf


def fusion_calibration(tr_df, te_df, eta_train, eta_test):
    """AUC(t), IBS and D-cal for the fusion Cox head via a TRAIN-ONLY Breslow baseline.
    Returns (auc_mean, ibs, d_cal_p) — any degenerate fold degrades to NaN, never raises."""
    t_tr, e_tr = _arr(tr_df, "time_months"), tr_df["event"].astype(int).values
    t_te, e_te = _arr(te_df, "time_months"), te_df["event"].astype(int).values
    auc = cumulative_dynamic_auc_mean(t_tr, e_tr, t_te, e_te, eta_test)
    grid = time_grid(t_tr, e_tr, t_te)
    ibs = float("nan"); dcal_p = float("nan")
    if grid is not None:
        H = breslow_baseline(t_tr, e_tr, eta_train, grid)
        S = cox_survival(eta_test, H)                       # (n_test, len grid)
        ibs = ipcw_ibs(t_tr, e_tr, t_te, e_te, S, grid)
    # S_i(t_i): Breslow H0 evaluated at each subject's observed time, then S=exp(-H0 e^eta)
    H_obs = breslow_baseline(t_tr, e_tr, eta_train, t_te)
    s_obs = np.exp(-H_obs * np.exp(np.asarray(eta_test, float)))
    dcal_p = d_calibration(s_obs, e_te).get("d_cal_p", float("nan"))
    return auc, ibs, dcal_p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=15, help="repeated patient-disjoint splits")
    ap.add_argument("--smoke", action="store_true",
                    help="fast validation: 3 splits, classical comparators only (no torch zoo)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = 3 if args.smoke else args.splits
    df, blocks, gene_df = load_patient_blocks()
    rich_cols = CLIN + [f"PC{i+1}" for i in range(N_PC)] + PROG + blocks["transferred"]
    print(f"[fusion] patients={len(df)} events={int(df['event'].sum())} modalities={list(blocks)} "
          f"transcriptome=in-fold PCA({N_PC}) over {gene_df.shape[1]} genes splits={R} dev={dev}", flush=True)

    dC, C_fus, C_clin = [], [], []
    AUC, IBS, DCAL = [], [], []
    zoo_C: dict[str, list] = {}          # comparator name -> per-split Uno-C
    dC_best = []                         # fusion - best non-fusion comparator, per split
    for r in range(R):
        rs = np.random.default_rng(SEED + r)
        ix = rs.permutation(len(df)); cut = int(0.7 * len(df)); vcut = int(0.85 * len(df))
        tr, va, te = ix[:cut], ix[cut:vcut], ix[vcut:]
        fdf = infold_transcriptome(df, gene_df, tr)          # in-fold transcriptome PCA
        tr_df, te_df = fdf.iloc[tr], fdf.iloc[te]
        if te_df["event"].sum() < 8:
            continue
        scaler = {m: (zfill(fdf.loc[tr, cols]).values.mean(0), zfill(fdf.loc[tr, cols]).values.std(0) + 1e-6)
                  for m, cols in blocks.items()}
        model = make_model(blocks, dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        btr = to_batch(fdf, blocks, tr, scaler, dev)
        ttr = torch.tensor(fdf.loc[tr, "time_months"].clip(lower=0.1).values, dtype=torch.float32, device=dev)
        etr = torch.tensor(fdf.loc[tr, "event"].astype(float).values, dtype=torch.float32, device=dev)
        bva = to_batch(fdf, blocks, va, scaler, dev)
        tva = torch.tensor(fdf.loc[va, "time_months"].clip(lower=0.1).values, dtype=torch.float32, device=dev)
        eva = torch.tensor(fdf.loc[va, "event"].astype(float).values, dtype=torch.float32, device=dev)
        best, best_state, bad = 1e9, None, 0
        for ep in range(300):
            model.train(); opt.zero_grad()
            loss = model.nll(btr, ttr, etr); loss.backward(); opt.step()
            if ep % 5 == 0:
                model.eval()
                with torch.no_grad():
                    vl = float(model.nll(bva, tva, eva))
                if vl < best - 1e-4: best, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
                else: bad += 1
                if bad >= 10: break
        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            eta_te = model(to_batch(fdf, blocks, te, scaler, dev))["risk"].cpu().numpy()
            eta_tr = model(to_batch(fdf, blocks, tr, scaler, dev))["risk"].cpu().numpy()

        c_fus = _uno(tr_df, te_df, eta_te)
        auc, ibs, dcal_p = fusion_calibration(tr_df, te_df, eta_tr, eta_te)
        # clinical-only Cox floor (shared PenalizedCox, train-only standardized)
        clin_est = PenalizedCox(CLIN).fit(tr_df)
        c_clin = _uno(tr_df, te_df, clin_est.risk(te_df))
        # the honest ML zoo on the SAME rich multimodal feature set
        comps = build_comparators(rich_cols, seed=SEED)
        if args.smoke:
            comps = [c for c in comps if "torch" not in c.name]  # skip neural comparators for speed
        best_other = float("nan")
        for comp in comps:
            try:
                cc = _uno(tr_df, te_df, comp.fit(tr_df).risk(te_df))
            except Exception:
                cc = float("nan")
            zoo_C.setdefault(comp.name, []).append(cc)
            if not comp.is_baseline and np.isfinite(cc):
                best_other = cc if not np.isfinite(best_other) else max(best_other, cc)

        C_fus.append(c_fus); C_clin.append(c_clin); dC.append(c_fus - c_clin)
        AUC.append(auc); IBS.append(ibs); DCAL.append(dcal_p)
        if np.isfinite(best_other) and np.isfinite(c_fus):
            dC_best.append(c_fus - best_other)

    dC = np.array(dC, float); dC_best = np.array(dC_best, float)
    beats_clinical = bool(np.isfinite(dC).any() and np.nanpercentile(dC, 2.5) > 0)
    beats_best_zoo = bool(dC_best.size and np.nanpercentile(dC_best, 2.5) > 0)
    zoo_leaderboard = {name: round(float(np.nanmean(v)), 4) for name, v in sorted(
        zoo_C.items(), key=lambda kv: -np.nanmean(kv[1]))}
    # pre-registered claim gating: OS licenses only a technical-validation claim
    endpoint_spec = {"name": "overall_survival", "endpoint_type": "overall_survival",
                     "role": "open_commpass_os_primary"}
    claim_flags = gate_claims(endpoint_spec, omics_increment_confirmed=beats_clinical,
                              external_validation_available=False)

    res = {
        "model": "ResistanceModel({clinical,transcriptome,programs,transferred}, CrossAttentionFusion, CoxHead)",
        "transcriptome_features": f"in-fold PCA({N_PC}) over gene_matrix.npz (train-only per split); omics.csv NOT used",
        "endpoint": "overall_survival (public MMRF-CoMMpass OPEN)",
        "n_patients": int(len(df)), "n_splits": int(dC.size), "device": dev,
        "metric": "Uno IPCW C-index (primary); AUC(t)/IBS/D-cal via train-only Breslow baseline",
        "C_fusion_mean": float(np.nanmean(C_fus)) if C_fus else float("nan"),
        "C_clinical_mean": float(np.nanmean(C_clin)) if C_clin else float("nan"),
        "auc_t_mean": float(np.nanmean(AUC)) if AUC else float("nan"),
        "ibs_mean": float(np.nanmean(IBS)) if IBS else float("nan"),
        "d_cal_p_mean": float(np.nanmean(DCAL)) if DCAL else float("nan"),
        "dC_fusion_vs_clinical_mean": float(np.nanmean(dC)) if dC.size else float("nan"),
        "dC_vs_clinical_ci95": [float(np.nanpercentile(dC, 2.5)), float(np.nanpercentile(dC, 97.5))] if dC.size else None,
        "dC_fusion_vs_best_comparator_mean": float(np.nanmean(dC_best)) if dC_best.size else float("nan"),
        "dC_vs_best_comparator_ci95": [float(np.nanpercentile(dC_best, 2.5)), float(np.nanpercentile(dC_best, 97.5))] if dC_best.size else None,
        "comparator_leaderboard_uno_c": zoo_leaderboard,
        "comparator_backends": backend_report(),
        "gate_fusion_beats_clinical": beats_clinical,
        "gate_fusion_beats_best_comparator": beats_best_zoo,
        "claim_gate": claim_flags,
        "note": ("ΔC vs the BEST classical comparator is the load-bearing test of 'multimodal fusion "
                 "adds discrimination'; a CI including 0 is a pre-registered NULL. On an OS endpoint the "
                 "claim gate permits only technical-validation language (no relapse/PFS/biological claim)."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fusion_model_value.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
