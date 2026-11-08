#!/usr/bin/env python
"""Fold-0 SMOKE GATE for Hierarchical Subtype Survival (HSS).

Pre-registered question (decided BEFORE the run):
  Does HSS beat an INDEPENDENT-per-subtype model on the SMALLEST cytogenetic
  subtype's IPCW integrated Brier score (IBS, lower=better), on one
  patient-disjoint fold (seed=42), for the Cox head?

GATE:  HSS_IBS <= INDEPENDENT_IBS on the smallest subtype  -> proceed to full impl.
       otherwise -> STOP and report the honest null. DO NOT tune to a target.

Self-contained: touches no existing pipeline module (the production code is only
modified AFTER this gate passes). Reuses the matched cohort + a patient-disjoint
hash split (the patient-aware splitter, not the row-indexed one).

Fixed hyperparameters (pre-registered, NOT swept here):
  trunk hidden=64, dropout=0.1; Adam lr=2e-3 wd=1e-4; epochs<=200 patience=30;
  distill lambda=1.0, start epoch 20 (on-policy teacher = detached agnostic head,
  fixed not EMA); curve-distill grid = train event-time deciles.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mm_tte_survival.data.cohort import build_matched_cohort
from mm_tte_survival.metrics import fast_c_index as cidx
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score

SEED = 42
HIDDEN, DROPOUT = 64, 0.1
LR, WD, EPOCHS, PATIENCE = 2e-3, 1e-4, 200, 30
LAMBDA_DISTILL, DISTILL_START = 1.0, 20
TIME, EVENT, IDC = "time_months", "event", "patient_id"

CFG = {"paths": {"clinical": str(ROOT / "data/real/clinical_survival.csv"),
                 "cytogenetics": str(ROOT / "data/real/cytogenetics.csv"),
                 "omics": str(ROOT / "data/real/omics.csv")},
       "schema": {"id_col": IDC, "time_col": TIME, "event_col": EVENT,
                  "clinical_cols": ["age", "sex_M", "iss_2", "iss_3", "line_of_therapy", "albumin", "b2m"],
                  "cytogenetic_cols": ["amp1q", "del1p", "del13q", "del17p", "t_4_14", "t_11_14", "t_14_16", "hyperdiploid"]}}


def hash_split(pid: str, seed: int = SEED) -> str:
    import hashlib
    v = int(hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if v < 0.70 else ("val" if v < 0.85 else "test")


def cox_nll(eta: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    eta = eta.reshape(-1)
    if e.sum() < 1:
        return eta.sum() * 0.0
    order = torch.argsort(t, descending=True)
    eta_o, e_o = eta[order], e[order].float()
    logcum = torch.logcumsumexp(eta_o, dim=0)
    return -((eta_o - logcum) * e_o).sum() / torch.clamp(e_o.sum(), min=1.0)


def breslow_baseline(t_tr, e_tr, eta_tr, grid):
    """H0(grid) Breslow estimator from train predictor (numpy)."""
    order = np.argsort(t_tr)
    t_s, e_s, r_s = t_tr[order], e_tr[order], np.exp(eta_tr[order])
    # risk set sum_{k: t_k >= t_j} exp(eta_k) via reverse cumsum
    rev_cum = np.cumsum(r_s[::-1])[::-1]
    h0 = np.zeros_like(t_s, dtype=float)
    h0[e_s == 1] = 1.0 / np.clip(rev_cum[e_s == 1], 1e-8, None)
    H_at_event = np.cumsum(h0)
    # step function: H0(tau) = sum of increments at event times <= tau
    H_grid = np.array([H_at_event[t_s <= tau][-1] if np.any(t_s <= tau) else 0.0 for tau in grid])
    return H_grid


def surv_curves(eta, H_grid):
    """S(tau|x) = exp(-H0(tau) * exp(eta)), shape (n, len(grid))."""
    return np.exp(-np.outer(np.exp(eta), H_grid))


def ibs(t_tr, e_tr, t_te, e_te, S_te, grid):
    surv_tr = Surv.from_arrays(e_tr.astype(bool), t_tr)
    surv_te = Surv.from_arrays(e_te.astype(bool), t_te)
    return float(integrated_brier_score(surv_tr, surv_te, S_te, grid))


# ---------------------------------------------------------------------------
class Trunk(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(),
                                 nn.Dropout(DROPOUT), nn.Linear(HIDDEN, HIDDEN), nn.GELU())

    def forward(self, x):
        return self.net(x)


class IndependentCox(nn.Module):
    """Baseline: trunk + single Cox head, trained on ONE subtype's patients only."""
    def __init__(self, d):
        super().__init__()
        self.trunk, self.head = Trunk(d), nn.Linear(HIDDEN, 1)

    def forward(self, x):
        return self.head(self.trunk(x)).reshape(-1)


class HSS(nn.Module):
    """Shared trunk + agnostic head + per-subtype heads + multi-label membership
    softmax mixer (agnostic always present; pure-agnostic fallback)."""
    def __init__(self, d, n_sub):
        super().__init__()
        self.trunk = Trunk(d)
        self.agnostic = nn.Linear(HIDDEN, 1)
        self.subs = nn.ModuleList([nn.Linear(HIDDEN, 1) for _ in range(n_sub)])
        self.gate = nn.Parameter(torch.zeros(n_sub + 1))   # logits: [agnostic, sub_0..]
        self.log_temp = nn.Parameter(torch.zeros(1))       # learnable softmax temperature

    def heads(self, z):
        eta_ag = self.agnostic(z).reshape(-1)
        eta_s = torch.stack([h(z).reshape(-1) for h in self.subs], dim=1)  # (n, n_sub)
        return eta_ag, eta_s

    def forward(self, z, m):
        """m: (n, n_sub) membership {0,1}. Returns mixed eta, agnostic eta, per-sub eta."""
        eta_ag, eta_s = self.heads(z)
        temp = torch.exp(self.log_temp) + 1e-3
        # build gate logits per patient: agnostic always; subtype k only if member
        present = torch.cat([torch.ones(m.shape[0], 1, device=m.device), m], dim=1)  # (n, 1+n_sub)
        logits = self.gate.unsqueeze(0) / temp
        logits = logits.masked_fill(present < 0.5, float("-inf"))
        w = torch.softmax(logits, dim=1)                                   # (n, 1+n_sub)
        eta_all = torch.cat([eta_ag.unsqueeze(1), eta_s], dim=1)           # (n, 1+n_sub)
        eta_mixed = (w * eta_all).sum(dim=1)
        return eta_mixed, eta_ag, eta_s


# ---------------------------------------------------------------------------
def load_fold():
    df, g = build_matched_cohort(CFG)
    sub_cols = g["cyto"]
    feat_cols = g["clinical"] + g["omics"]
    df["split"] = df[IDC].astype(str).map(hash_split)
    for c in feat_cols + sub_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    tr = df.split.eq("train").values
    # train-only median impute + standardize
    X = df[feat_cols].copy()
    X = X.fillna(X[tr].median()).fillna(0.0)
    X = (X - X[tr].mean()) / X[tr].std().replace(0, 1)
    X = X.fillna(0.0).values.astype("float32")
    M = (df[sub_cols].fillna(0).values > 0).astype("float32")
    t = pd.to_numeric(df[TIME]).clip(lower=0.1).values.astype("float32")
    e = df[EVENT].astype(int).values
    sp = df.split.values
    return X, M, t, e, sp, sub_cols


def grid_for(t_tr, e_tr, t_te):
    ev = t_tr[e_tr == 1]
    lo = max(np.quantile(ev, 0.1), t_te.min() + 1e-3, 1.0)
    hi = min(t_tr.max(), t_te.max()) - 1e-3
    if hi <= lo:
        return None
    return np.linspace(lo, hi, 12)


def train_independent(Xtr, ttr, etr, Xva, tva, eva):
    """Baseline trained on ONE subtype's patients. To be a FAIR (strong) baseline
    it always gets an early-stopping signal: the subtype's own val patients if
    enough events exist, else an internal 20% hold-out of its train fold."""
    torch.manual_seed(SEED)
    if len(Xva) >= 5 and eva.sum() >= 2:
        Xfit, tfit, efit, Xes, tes, ees = Xtr, ttr, etr, Xva, tva, eva
    else:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(Xtr)); k = max(1, int(0.2 * len(Xtr)))
        es, fit = perm[:k], perm[k:]
        Xfit, tfit, efit = Xtr[fit], ttr[fit], etr[fit]
        Xes, tes, ees = Xtr[es], ttr[es], etr[es]
    m = IndependentCox(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    xt, tt, et = torch.tensor(Xfit), torch.tensor(tfit), torch.tensor(efit.astype("float32"))
    best, best_ep, best_state = -np.inf, -1, None
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad()
        loss = cox_nll(m(xt), tt, et); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            cv = cidx(tes, ees, m(torch.tensor(Xes)).numpy())
        if np.isfinite(cv) and cv > best:
            best, best_ep = cv, ep
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        elif ep - best_ep >= PATIENCE:
            break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    return m


def train_hss(Xtr, Mtr, ttr, etr, Xva, Mva, tva, eva):
    torch.manual_seed(SEED)
    n_sub = Mtr.shape[1]
    m = HSS(Xtr.shape[1], n_sub)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    xt, Mt = torch.tensor(Xtr), torch.tensor(Mtr)
    tt, et = torch.tensor(ttr), torch.tensor(etr.astype("float32"))
    xv, Mv = torch.tensor(Xva), torch.tensor(Mva)
    grid = grid_for(ttr, etr, ttr)
    gt = torch.tensor(grid, dtype=torch.float32) if grid is not None else None
    best, best_ep, best_state = -np.inf, -1, None
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad()
        z = m.trunk(xt)
        eta_mix, eta_ag, eta_s = m(z, Mt)
        loss = cox_nll(eta_mix, tt, et) + cox_nll(eta_ag, tt, et)
        for s in range(n_sub):
            mask = Mt[:, s] > 0.5
            if mask.sum() >= 3 and et[mask].sum() >= 1:
                loss = loss + cox_nll(eta_s[mask, s], tt[mask], et[mask])
        # cross-head survival-CURVE distillation: subtype head -> detached agnostic
        if ep >= DISTILL_START and gt is not None:
            with torch.no_grad():
                H = torch.tensor(breslow_baseline(ttr, etr, eta_ag.detach().numpy(), grid),
                                 dtype=torch.float32)
            dl = torch.tensor(0.0)
            for s in range(n_sub):
                mask = Mt[:, s] > 0.5
                if mask.sum() < 3:
                    continue
                Ss = torch.exp(-torch.exp(eta_s[mask, s]).unsqueeze(1) * H.unsqueeze(0))
                S0 = torch.exp(-torch.exp(eta_ag[mask].detach()).unsqueeze(1) * H.unsqueeze(0))
                dl = dl + ((Ss - S0) ** 2).mean()
            loss = loss + LAMBDA_DISTILL * dl
        loss.backward(); opt.step()
        # early stop on val mixed-predictor C-index
        if eva.sum() >= 2:
            m.eval()
            with torch.no_grad():
                cv = cidx(tva, eva, m(m.trunk(xv), Mv)[0].numpy())
            if np.isfinite(cv) and cv > best:
                best, best_ep = cv, ep
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
            elif ep - best_ep >= PATIENCE:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    return m


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    X, M, t, e, sp, sub_cols = load_fold()
    tr, va, te = sp == "train", sp == "val", sp == "test"
    print(f"matched cohort N={len(X)}  train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"(events test={e[te].sum()})\n")

    # smallest subtype by TRAIN membership (decided by data, not by outcome)
    train_counts = {c: int(M[tr, i].sum()) for i, c in enumerate(sub_cols)}
    print("train subtype membership:", dict(sorted(train_counts.items(), key=lambda kv: kv[1])))
    smallest = min(train_counts, key=train_counts.get)
    print(f"\n>>> smallest subtype (pre-registered anchor): {smallest} "
          f"(train n={train_counts[smallest]})\n")

    # train HSS once (joint)
    hss = train_hss(X[tr], M[tr], t[tr], e[tr], X[va], M[va], t[va], e[va])

    rows = []
    for i, s in enumerate(sub_cols):
        mem = M[:, i] > 0.5
        s_tr, s_va, s_te = tr & mem, va & mem, te & mem
        if s_te.sum() < 3 or e[s_te].sum() < 1 or s_tr.sum() < 8:
            rows.append({"subtype": s, "test_n": int(s_te.sum()), "test_ev": int(e[s_te].sum()),
                         "note": "too few test patients/events"})
            continue
        grid = grid_for(t[s_tr], e[s_tr], t[s_te])
        if grid is None:
            rows.append({"subtype": s, "test_n": int(s_te.sum()), "note": "no valid IBS grid"})
            continue

        # INDEPENDENT: trunk+head trained on subtype-s patients only
        ind = train_independent(X[s_tr], t[s_tr], e[s_tr], X[s_va], t[s_va], e[s_va])
        with torch.no_grad():
            eta_tr_i = ind(torch.tensor(X[s_tr])).numpy()
            eta_te_i = ind(torch.tensor(X[s_te])).numpy()
        H_i = breslow_baseline(t[s_tr], e[s_tr], eta_tr_i, grid)
        ibs_ind = ibs(t[s_tr], e[s_tr], t[s_te], e[s_te], surv_curves(eta_te_i, H_i), grid)
        c_ind = cidx(t[s_te], e[s_te], eta_te_i)

        # HSS: mixed predictor restricted to subtype-s patients (Breslow on subtype-s train)
        with torch.no_grad():
            eta_tr_h = hss(hss.trunk(torch.tensor(X[s_tr])), torch.tensor(M[s_tr]))[0].numpy()
            eta_te_h = hss(hss.trunk(torch.tensor(X[s_te])), torch.tensor(M[s_te]))[0].numpy()
        H_h = breslow_baseline(t[s_tr], e[s_tr], eta_tr_h, grid)
        ibs_hss = ibs(t[s_tr], e[s_tr], t[s_te], e[s_te], surv_curves(eta_te_h, H_h), grid)
        c_hss = cidx(t[s_te], e[s_te], eta_te_h)

        rows.append({"subtype": s, "test_n": int(s_te.sum()), "test_ev": int(e[s_te].sum()),
                     "ibs_independent": round(ibs_ind, 4), "ibs_hss": round(ibs_hss, 4),
                     "ibs_delta(hss-ind)": round(ibs_hss - ibs_ind, 4),
                     "c_independent": round(c_ind, 3), "c_hss": round(c_hss, 3),
                     "hss_better_ibs": bool(ibs_hss <= ibs_ind)})

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))

    anchor = res[res.subtype == smallest].iloc[0]
    print("\n" + "=" * 64)
    if "ibs_hss" not in anchor or pd.isna(anchor.get("ibs_hss", np.nan)):
        print(f"GATE INCONCLUSIVE: smallest subtype {smallest} too small for IBS.")
        print("Reporting honestly; not proceeding on an inconclusive anchor.")
        return
    verdict = "PASS" if anchor["hss_better_ibs"] else "FAIL"
    print(f"GATE [{verdict}] smallest subtype={smallest}: "
          f"IBS independent={anchor['ibs_independent']} vs HSS={anchor['ibs_hss']} "
          f"(Δ={anchor['ibs_delta(hss-ind)']}, lower=better)")
    if verdict == "FAIL":
        print("STOP per pre-registration: HSS did not beat independent on the smallest "
              "subtype's IBS. Reporting the honest null. NOT tuning to a target.")
    else:
        print("Proceed to full file-by-file implementation (gate passed on fold 0).")
    print("=" * 64)


if __name__ == "__main__":
    main()
