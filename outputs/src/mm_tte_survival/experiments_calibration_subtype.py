"""Optional pre-registered calibration one-shot (see docs/FRAMING_SUBTYPE_AWARE_NULL.md §4).

Stage D / Direction-2 tested whether subtype conditioning improves per-subtype risk
(discrimination + IBS). This asks a DIFFERENT question, exactly once: does conditioning
the *calibration* on subtype help? We hold the risk model fixed (pooled penalised Cox)
and only let the BASELINE HAZARD be subtype-stratified, then measure calibration
(IPCW-IBS + D-calibration) under three baselines on identical folds:

  pooled     - one Breslow baseline for everyone
  real       - Breslow baseline stratified by the real subtype
  scramble   - Breslow baseline stratified by PERMUTED subtype (negative control)

PRE-REGISTERED hard stop: a promotable calibration claim requires BOTH (a) real beats
pooled AND beats scramble by the margin, AND (b) external replication on a FISH cohort
with survival. GSE19784 carries NO survival in the open record, so (b) is unmeetable on
open data -> this test is reported as CHARACTERIZATION / a strengthened negative,
never a promotable claim, regardless of (a). No new architecture is built.

Endpoint = OS technical validation. No relapse/PFS or clinical-use claims.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

from .config import ensure_outdir
from .data.cohort import build_matched_cohort, build_membership_matrix
from .data.gene_expression import load_gene_matrix
from .data.omics_pca import OmicsInFoldPCA
from .data.splits import patient_disjoint_stratified_split
from .survival_curves import breslow_baseline, cox_survival, ipcw_ibs, time_grid

TIME, EVENT = "time_months", "event"
MARGIN = 0.01  # pre-registered minimum real-vs-(pooled/scramble) IBS improvement


def _surv_at_time(t_tr, e_tr, eta_tr, t_te, eta_te):
    """Per-patient predicted S(t_i) at each test patient's own observed time."""
    H = breslow_baseline(t_tr, e_tr, eta_tr, np.asarray(t_te))  # H0 at each test time
    return np.exp(-H * np.exp(np.asarray(eta_te)))


def d_calibration(S_at_t, e, n_bins: int = 10):
    """Censoring-aware D-calibration (Haider et al. 2020). Returns (chi2, p).
    Well-calibrated => S(T) ~ Uniform(0,1) => high p (close to uniform bins)."""
    S = np.clip(np.asarray(S_at_t, float), 1e-8, 1.0)
    e = np.asarray(e, int)
    edges = np.linspace(0, 1, n_bins + 1)
    counts = np.zeros(n_bins)
    for s, ev in zip(S, e):
        if ev == 1:
            b = min(int(s * n_bins), n_bins - 1)
            counts[b] += 1.0
        else:
            # censored: distribute remaining mass uniformly over [0, s]
            for b in range(n_bins):
                lo, hi = edges[b], edges[b + 1]
                if hi <= s:
                    counts[b] += (1.0 / n_bins) / s
                elif lo < s:
                    counts[b] += (s - lo) / s
    n = counts.sum()
    if n <= 0:
        return float("nan"), float("nan")
    expected = n / n_bins
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=n_bins - 1))
    return chi2, p


def run_subtype_calibration(cfg: dict) -> dict:
    outdir = ensure_outdir(cfg["paths"].get("outdir", "outputs/calibration_subtype"))
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
        cph = CoxPHFitter(penalizer=0.1).fit(d[tr], duration_col=TIME, event_col=EVENT)
        eta = cph.predict_log_partial_hazard(d[cols]).values  # fixed risk for all schemes

        rng = np.random.default_rng(base_seed + f)
        M_scr = M_real[rng.permutation(len(M_real))]
        for i, s in enumerate(sub_cols):
            for scheme, Msrc in [("pooled", None), ("real", M_real), ("scramble", M_scr)]:
                mem_eval = M_real[:, i] > 0.5            # evaluate on the SAME real-subtype patients
                s_te = (~tr) & mem_eval
                if s_te.sum() < 3 or e[s_te].sum() < 1:
                    continue
                if scheme == "pooled":
                    s_tr = tr                            # baseline from all train
                else:
                    s_tr = tr & (Msrc[:, i] > 0.5)       # baseline from (real/scrambled) subtype train
                if s_tr.sum() < 8:
                    continue
                grid = time_grid(t[s_tr], e[s_tr], t[s_te])
                if grid is None:
                    continue
                H = breslow_baseline(t[s_tr], e[s_tr], eta[s_tr], grid)
                ibs = ipcw_ibs(t[s_tr], e[s_tr], t[s_te], e[s_te], cox_survival(eta[s_te], H), grid)
                S_at = _surv_at_time(t[s_tr], e[s_tr], eta[s_tr], t[s_te], eta[s_te])
                _, dcal_p = d_calibration(S_at, e[s_te])
                rows.append({"fold": f, "subtype": s, "scheme": scheme,
                             "test_n": int(s_te.sum()), "ibs": round(float(ibs), 4),
                             "dcal_p": round(float(dcal_p), 4)})

    detail = pd.DataFrame(rows)
    detail.to_csv(outdir / "calibration_subtype_detail.csv", index=False)
    summ = (detail.dropna(subset=["ibs"]).groupby(["subtype", "scheme"])
            .agg(n_folds=("fold", "nunique"), mean_ibs=("ibs", "mean"),
                 mean_dcal_p=("dcal_p", "mean")).reset_index().round(4))
    summ.to_csv(outdir / "calibration_subtype_summary.csv", index=False)
    decision = _decide(summ, M_real, sub_cols)
    _write(outdir, summ, decision)
    return {"outdir": str(outdir), "summary": summ, "decision": decision}


def _decide(summ, M_real, sub_cols) -> dict:
    prev = {c: float(M_real[:, i].mean()) for i, c in enumerate(sub_cols)}
    small = [c for c, _ in sorted(prev.items(), key=lambda kv: kv[1])[:2]]
    piv = summ[summ.subtype.isin(small)].groupby("scheme")["mean_ibs"].mean()
    pooled = float(piv.get("pooled", np.nan))
    real = float(piv.get("real", np.nan))
    scramble = float(piv.get("scramble", np.nan))
    beats_pooled = (pooled - real) > MARGIN
    beats_scramble = (scramble - real) > MARGIN
    internal_pass = bool(beats_pooled and beats_scramble)
    # Pre-registered: external replication (FISH cohort with survival) is UNMEETABLE on
    # open data (GSE19784 has no survival) -> never promotable, whatever the internal result.
    if internal_pass:
        verdict = (
            "CHARACTERIZATION ONLY — internal calibration signal present: subtype-stratified "
            "baseline beats BOTH pooled and scramble on small-subtype IBS (del17p-driven). "
            "This is NOT a null. But the pre-registered external-replication requirement is "
            "UNMEETABLE on open data (no survival in GSE19784), so it CANNOT be promoted to a "
            "claim; reported as an unverifiable internal positive, hypothesis-generating only.")
    else:
        verdict = (
            "STRENGTHENED NEGATIVE — subtype-stratified calibration does NOT beat pooled+scramble "
            "on small-subtype IBS, and external replication is unmeetable on open data. Stop.")
    return {"small_subtypes": small, "small_subtype_mean_ibs": piv.round(4).to_dict(),
            "pooled": round(pooled, 4), "real": round(real, 4), "scramble": round(scramble, 4),
            "margin": MARGIN, "internal_pass": internal_pass,
            "promotable": False, "verdict": verdict}


def _write(outdir, summ, decision) -> None:
    lines = ["# Subtype-conditioned calibration — pre-registered one-shot (OS technical validation)", "",
             f"Small subtypes: {decision['small_subtypes']}", "",
             "Mean small-subtype IPCW-IBS by baseline scheme (lower=better):"]
    for sch, v in decision["small_subtype_mean_ibs"].items():
        lines.append(f"  - {sch}: {v}")
    lines += ["",
              f"pooled-real = {decision['pooled'] - decision['real']:+.4f}, "
              f"scramble-real = {decision['scramble'] - decision['real']:+.4f} "
              f"(margin {decision['margin']}; +ve favours real subtype stratification).", "",
              f"## VERDICT (promotable={decision['promotable']})",
              decision["verdict"], "",
              "- D-calibration p per subtype/scheme in calibration_subtype_summary.csv "
              "(higher p = better-calibrated).",
              "- Endpoint = OS technical validation. NO relapse/PFS claim; NO clinical-use claim."]
    (outdir / "calibration_subtype_decision.md").write_text("\n".join(lines) + "\n")
