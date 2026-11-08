"""Direction-2 probe — is ANY neural regularization useful, or is penalised Cox
the honest answer?

Stage D compared HSS to INDEPENDENT-per-subtype Cox (a weak, unstable baseline).
The fair test of "does the neural shared representation help" is against the
STRONG simple baseline: a pooled penalised Cox. We compute per-subtype IPCW-IBS
for three models with a COMMON subtype-calibrated Breslow baseline (so the only
thing that differs is the risk-ranking each model produces):

  independent_cox  - lifelines Cox fit on that subtype's patients only
  pooled_cox       - lifelines penalised Cox fit on ALL train (the strong baseline)
  pooled_neural    - shared-trunk MLP-Cox (HSS with n_subtypes=0) fit on ALL train

Pre-registered null: if pooled_neural does NOT beat pooled_cox on small-subtype
IBS by a margin, the regularization angle is ALSO null -> penalised Cox is the
answer (consistent with the original review). Leak-free: in-fold PCA + CNV-only.

Endpoint = OS technical validation. No relapse/PFS or clinical-use claims.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .config import ensure_outdir
from .data.cohort import build_matched_cohort, build_membership_matrix
from .data.gene_expression import load_gene_matrix
from .data.omics_pca import OmicsInFoldPCA
from .data.splits import patient_disjoint_stratified_split
from .metrics import fast_c_index
from .survival_curves import breslow_baseline, cox_survival, ipcw_ibs, time_grid
from .training.trainer_hss import train_single, _eta

TIME, EVENT = "time_months", "event"


def _ibs_from_eta(eta_tr, eta_te, t_sr, e_sr, t_ste, e_ste):
    """Common subtype-calibrated Breslow baseline -> isolates risk-ranking quality."""
    grid = time_grid(t_sr, e_sr, t_ste)
    if grid is None:
        return np.nan, np.nan
    H = breslow_baseline(t_sr, e_sr, eta_tr, grid)
    ibs = ipcw_ibs(t_sr, e_sr, t_ste, e_ste, cox_survival(eta_te, H), grid)
    return float(ibs), float(fast_c_index(t_ste, e_ste, eta_te))


def run_regularization(cfg: dict) -> dict:
    outdir = ensure_outdir(cfg["paths"].get("outdir", "outputs/regularization"))
    cache = Path(cfg["paths"].get("gene_matrix", "data/real/gene_matrix.npz"))
    n_folds = int(cfg.get("validation", {}).get("folds", 5))
    base_seed = int(cfg.get("seed", 42))
    k = int(cfg.get("features", {}).get("max_omics_features", 16))
    n_topvar = int(cfg.get("features", {}).get("n_topvar", 2000))

    cfg = {**cfg, "features": {**cfg.get("features", {}), "max_omics_features": k}}
    df, g = build_matched_cohort(cfg)
    gene_df = load_gene_matrix(cache)
    df = df[df["patient_id"].astype(str).isin(set(gene_df.index))].reset_index(drop=True)
    clin, sub_cols = g["clinical"], g["cyto"]
    M = build_membership_matrix(df, sub_cols)
    t = pd.to_numeric(df[TIME]).clip(lower=0.1).values.astype("float32")
    e = df[EVENT].astype(int).values
    ids = df["patient_id"].astype(str).values

    rows = []
    for f in range(n_folds):
        split = patient_disjoint_stratified_split(df, "patient_id", EVENT, base_seed + f)
        tr = split.values
        # leak-free features, standardized on all-train
        infold = OmicsInFoldPCA(k=k, n_topvar=n_topvar, seed=base_seed).fit_transform(gene_df, ids[tr]).reindex(ids)
        infold.columns = [f"inPC{i+1}" for i in range(infold.shape[1])]
        feat = pd.concat([df[clin].reset_index(drop=True), infold.reset_index(drop=True)], axis=1)
        feat = feat.apply(pd.to_numeric, errors="coerce")
        feat = feat.fillna(feat[tr].median()).fillna(0.0)
        feat = ((feat - feat[tr].mean()) / feat[tr].std().replace(0, 1)).fillna(0.0)
        cols = list(feat.columns)
        Xnp = feat.values.astype("float32")

        d = feat.copy(); d[TIME] = np.clip(t, 0.1, None); d[EVENT] = e.astype(int)
        cph_pool = CoxPHFitter(penalizer=0.1).fit(d[tr], duration_col=TIME, event_col=EVENT)
        neural = train_single(Xnp[tr], t[tr], e[tr],
                              np.zeros((0, Xnp.shape[1]), "float32"), t[:0], e[:0], cfg)

        for i, s in enumerate(sub_cols):
            mem = M[:, i] > 0.5
            s_tr, s_te = tr & mem, (~tr) & mem
            if s_te.sum() < 3 or e[s_te].sum() < 1 or s_tr.sum() < 8:
                continue
            etas = {}
            # independent cox (fit on subtype train only)
            try:
                ci = CoxPHFitter(penalizer=0.1).fit(d[s_tr], duration_col=TIME, event_col=EVENT)
                etas["independent_cox"] = (ci.predict_log_partial_hazard(d.loc[s_tr, cols]).values,
                                           ci.predict_log_partial_hazard(d.loc[s_te, cols]).values)
            except Exception:
                pass
            etas["pooled_cox"] = (cph_pool.predict_log_partial_hazard(d.loc[s_tr, cols]).values,
                                  cph_pool.predict_log_partial_hazard(d.loc[s_te, cols]).values)
            etas["pooled_neural"] = (_eta(neural, Xnp[s_tr], np.zeros((int(s_tr.sum()), 0), "float32")),
                                     _eta(neural, Xnp[s_te], np.zeros((int(s_te.sum()), 0), "float32")))
            for model, (etr, ete) in etas.items():
                ibs, c = _ibs_from_eta(etr, ete, t[s_tr], e[s_tr], t[s_te], e[s_te])
                rows.append({"fold": f, "subtype": s, "model": model, "test_n": int(s_te.sum()),
                             "ibs": round(ibs, 4), "cindex": round(c, 3)})

    detail = pd.DataFrame(rows)
    detail.to_csv(outdir / "regularization_detail.csv", index=False)
    summ = (detail.dropna(subset=["ibs"]).groupby(["subtype", "model"])
            .agg(n_folds=("fold", "nunique"), mean_test_n=("test_n", "mean"),
                 mean_ibs=("ibs", "mean"), mean_cindex=("cindex", "mean")).reset_index().round(4))
    summ.to_csv(outdir / "regularization_summary.csv", index=False)
    decision = _decide(summ, M, sub_cols)
    _write(outdir, summ, decision)
    return {"outdir": str(outdir), "summary": summ, "decision": decision}


def _decide(summ, M, sub_cols):
    prev = {c: float(M[:, i].mean()) for i, c in enumerate(sub_cols)}
    small = [c for c, _ in sorted(prev.items(), key=lambda kv: kv[1])[:2]]
    piv = (summ[summ.subtype.isin(small)].groupby("model")["mean_ibs"].mean())
    pn, pc = float(piv.get("pooled_neural", np.nan)), float(piv.get("pooled_cox", np.nan))
    margin = 0.01
    neural_beats_cox = (pc - pn) > margin   # lower IBS = better
    verdict = ("Neural regularization helps: pooled_neural beats pooled_cox on small-subtype IBS."
               if neural_beats_cox else
               "NULL: pooled_neural ~ pooled_cox on small-subtype IBS -> penalised Cox is the "
               "honest answer; no neural advantage (regularization angle does not survive).")
    return {"small_subtypes": small, "small_subtype_mean_ibs": piv.round(4).to_dict(),
            "pooled_neural": round(pn, 4), "pooled_cox": round(pc, 4),
            "pooled_cox_minus_neural": round(pc - pn, 4), "margin": margin,
            "neural_beats_cox": bool(neural_beats_cox), "verdict": verdict}


def _write(outdir: Path, summ, decision):
    lines = ["# Direction-2 — regularization probe (pooled neural vs pooled penalised Cox)", "",
             f"Small subtypes: {decision['small_subtypes']}", "",
             "Mean small-subtype IPCW-IBS (lower=better):"]
    for m, v in decision["small_subtype_mean_ibs"].items():
        lines.append(f"  - {m}: {v}")
    lines += ["",
              f"pooled_cox - pooled_neural = {decision['pooled_cox_minus_neural']:+.4f} "
              f"(margin {decision['margin']}; +ve favours neural).", "",
              f"## VERDICT: {'NEURAL HELPS' if decision['neural_beats_cox'] else 'NULL (use penalised Cox)'}",
              decision["verdict"], "",
              "- Endpoint = OS technical validation. NO relapse/PFS claim; NO clinical-use claim."]
    (outdir / "regularization_decision.md").write_text("\n".join(lines) + "\n")
