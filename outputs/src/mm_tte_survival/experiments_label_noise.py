"""Label-noise robustness — is the subtype-aware NULL an artifact of imperfect labels?

The standing objection to any subtype result on sequencing-inferred labels is
"your calls might be wrong." We answer it without FISH: corrupt each CNV subtype's
membership at its PUBLISHED sequencing-vs-FISH discordance rate (two-sided flip)
over many random draws, and re-run the Direction-2 comparison (pooled penalised
Cox vs independent-per-subtype Cox, per-subtype IPCW-IBS, common Breslow). If the
verdict — pooled penalised Cox is no worse than subtype-specific modelling, i.e.
no subtype-aware advantage — is preserved across the noise draws, then the NULL is
robust to label error at realistic rates.

Discordance rates are taken from the literature concordance table
(docs/literature_cnv_fish_concordance.md): del(17p) under-calls (~0.11),
amp(1q)/del(13) ~0.30, hyperdiploid ~0.10, del(1p) ~0.30.

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
from .experiments_regularization import _ibs_from_eta

TIME, EVENT = "time_months", "event"

# Published sequencing-vs-FISH discordance (symmetric flip) per CNV subtype.
FLIP_RATE = {"del17p": 0.11, "amp1q": 0.29, "del13q": 0.30, "del1p": 0.30, "hyperdiploid": 0.10}


def _flip(col: np.ndarray, rate: float, rng) -> np.ndarray:
    flip = rng.random(len(col)) < rate
    out = col.copy()
    out[flip] = 1.0 - out[flip]
    return out


def run_label_noise(cfg: dict) -> dict:
    outdir = ensure_outdir(cfg["paths"].get("outdir", "outputs/label_noise"))
    cache = Path(cfg["paths"].get("gene_matrix", "data/real/gene_matrix.npz"))
    n_folds = int(cfg.get("validation", {}).get("folds", 5))
    n_draws = int(cfg.get("validation", {}).get("noise_draws", 20))
    base_seed = int(cfg.get("seed", 42))
    k = int(cfg.get("features", {}).get("max_omics_features", 16))
    n_topvar = int(cfg.get("features", {}).get("n_topvar", 2000))

    cfg = {**cfg, "features": {**cfg.get("features", {}), "max_omics_features": k}}
    df, g = build_matched_cohort(cfg)
    gene_df = load_gene_matrix(cache)
    df = df[df["patient_id"].astype(str).isin(set(gene_df.index))].reset_index(drop=True)
    clin, sub_cols = g["clinical"], g["cyto"]
    M_real = build_membership_matrix(df, sub_cols)
    t = pd.to_numeric(df[TIME]).clip(lower=0.1).values.astype("float32")
    e = df[EVENT].astype(int).values
    ids = df["patient_id"].astype(str).values

    rows = []
    for f in range(n_folds):
        split = patient_disjoint_stratified_split(df, "patient_id", EVENT, base_seed + f)
        tr = split.values
        infold = OmicsInFoldPCA(k=k, n_topvar=n_topvar, seed=base_seed).fit_transform(gene_df, ids[tr]).reindex(ids)
        infold.columns = [f"inPC{i+1}" for i in range(infold.shape[1])]
        feat = pd.concat([df[clin].reset_index(drop=True), infold.reset_index(drop=True)], axis=1)
        feat = feat.apply(pd.to_numeric, errors="coerce")
        feat = feat.fillna(feat[tr].median()).fillna(0.0)
        feat = ((feat - feat[tr].mean()) / feat[tr].std().replace(0, 1)).fillna(0.0)
        cols = list(feat.columns)
        d = feat.copy(); d[TIME] = np.clip(t, 0.1, None); d[EVENT] = e.astype(int)
        cph_pool = CoxPHFitter(penalizer=0.1).fit(d[tr], duration_col=TIME, event_col=EVENT)
        eta_pool = cph_pool.predict_log_partial_hazard(d[cols]).values  # all patients

        # draw 0 = real labels; draws 1..n_draws = flipped at published rates
        for draw in range(n_draws + 1):
            rng = np.random.default_rng(base_seed + 7919 * f + draw)
            M = M_real.copy()
            if draw > 0:
                for i, s in enumerate(sub_cols):
                    M[:, i] = _flip(M_real[:, i], FLIP_RATE.get(s, 0.0), rng)
            for i, s in enumerate(sub_cols):
                mem = M[:, i] > 0.5
                s_tr, s_te = tr & mem, (~tr) & mem
                if s_te.sum() < 3 or e[s_te].sum() < 1 or s_tr.sum() < 8:
                    continue
                ibs_pool, c_pool = _ibs_from_eta(eta_pool[s_tr], eta_pool[s_te],
                                                 t[s_tr], e[s_tr], t[s_te], e[s_te])
                try:
                    ci = CoxPHFitter(penalizer=0.1).fit(d[s_tr], duration_col=TIME, event_col=EVENT)
                    ei_tr = ci.predict_log_partial_hazard(d.loc[s_tr, cols]).values
                    ei_te = ci.predict_log_partial_hazard(d.loc[s_te, cols]).values
                    ibs_ind, c_ind = _ibs_from_eta(ei_tr, ei_te, t[s_tr], e[s_tr], t[s_te], e[s_te])
                except Exception:
                    ibs_ind, c_ind = np.nan, np.nan
                rows.append({"fold": f, "draw": draw, "kind": "real" if draw == 0 else "flipped",
                             "subtype": s, "test_n": int(s_te.sum()),
                             "ibs_pooled": ibs_pool, "ibs_independent": ibs_ind,
                             "pooled_minus_independent": (ibs_pool - ibs_ind)
                             if np.isfinite(ibs_ind) else np.nan})

    detail = pd.DataFrame(rows)
    detail.to_csv(outdir / "label_noise_detail.csv", index=False)
    decision = _decide(detail, sub_cols)
    _write(outdir, decision)
    summ = decision["per_subtype"]
    return {"outdir": str(outdir), "summary": pd.DataFrame(summ).T, "decision": decision}


def _decide(detail: pd.DataFrame, sub_cols) -> dict:
    """For each subtype: real pooled IBS, flipped pooled IBS mean±sd, and whether
    'pooled not worse than independent' (gap <= margin) holds across flips."""
    margin = 0.01
    per = {}
    for s in sub_cols:
        real = detail[(detail.subtype == s) & (detail.kind == "real")]
        flip = detail[(detail.subtype == s) & (detail.kind == "flipped")]
        if real.empty:
            continue
        gaps = flip["pooled_minus_independent"].dropna()
        # verdict per draw: pooled is NOT worse than independent (gap <= margin)
        pooled_not_worse = float((gaps <= margin).mean()) if len(gaps) else np.nan
        per[s] = {
            "real_ibs_pooled": round(float(real["ibs_pooled"].mean()), 4),
            "flip_ibs_pooled_mean": round(float(flip["ibs_pooled"].mean()), 4) if len(flip) else np.nan,
            "flip_ibs_pooled_sd": round(float(flip["ibs_pooled"].std()), 4) if len(flip) else np.nan,
            "real_pooled_minus_independent": round(float(real["pooled_minus_independent"].mean()), 4),
            "flip_pooled_minus_independent_mean": round(float(gaps.mean()), 4) if len(gaps) else np.nan,
            "frac_draws_pooled_not_worse": round(pooled_not_worse, 3) if np.isfinite(pooled_not_worse) else np.nan,
        }
    # global verdict: in >=80% of (subtype,draw) the pooled Cox is not worse
    fracs = [v["frac_draws_pooled_not_worse"] for v in per.values()
             if v["frac_draws_pooled_not_worse"] == v["frac_draws_pooled_not_worse"]]
    robust = bool(fracs) and float(np.mean(fracs)) >= 0.80
    verdict = ("ROBUST: under realistic label noise, pooled penalised Cox remains no worse "
               "than subtype-specific modelling -> the subtype-aware NULL is not a label-noise "
               "artifact." if robust else
               "FRAGILE: the pooled-vs-subtype verdict flips under label noise -> per-subtype "
               "conclusions depend on label accuracy; interpret with caution.")
    return {"margin": margin, "mean_frac_pooled_not_worse": round(float(np.mean(fracs)), 3) if fracs else np.nan,
            "robust": robust, "verdict": verdict, "per_subtype": per}


def _write(outdir: Path, decision: dict) -> None:
    lines = ["# Label-noise robustness (OS technical validation)", "",
             "CNV subtype labels flipped at published sequencing-vs-FISH discordance rates "
             "(del17p 0.11, amp1q 0.29, del13q/del1p 0.30, hyperdiploid 0.10).", "",
             "Per subtype: pooled penalised-Cox IPCW-IBS under real vs flipped labels, and the "
             "fraction of noise draws in which pooled Cox is NOT worse than subtype-specific Cox.",
             ""]
    hdr = ["subtype", "real_ibs_pooled", "flip_ibs_pooled_mean", "flip_ibs_pooled_sd",
           "frac_draws_pooled_not_worse"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("| " + " | ".join("---" for _ in hdr) + " |")
    for s, v in decision["per_subtype"].items():
        lines.append("| " + " | ".join([s] + [str(v[c]) for c in hdr[1:]]) + " |")
    lines += ["",
              f"Mean fraction pooled-not-worse across subtypes = {decision['mean_frac_pooled_not_worse']} "
              f"(robust threshold 0.80).", "",
              f"## VERDICT: {'ROBUST' if decision['robust'] else 'FRAGILE'}",
              decision["verdict"], "",
              "- Endpoint = OS technical validation. NO relapse/PFS claim; NO clinical-use claim."]
    (outdir / "label_noise_decision.md").write_text("\n".join(lines) + "\n")
