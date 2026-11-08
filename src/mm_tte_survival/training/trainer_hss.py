"""Train Hierarchical Subtype Survival (HSS) and its architecture-matched baselines.

Baselines reuse the SAME model class with n_subtypes=0 (trunk + agnostic head),
so {independent, pooled, HSS} differ only in subtype structure + distillation,
never in capacity. Cox head is the validated path. No test set is touched during
training or selection (the old per-epoch `test_cindex_shadow` peeking is gone).
"""
from __future__ import annotations

import numpy as np
import torch

from ..metrics import fast_c_index
from ..models.hierarchical import (HierarchicalSubtypeSurvival, make_ema_teacher,
                                    update_ema_agnostic)
from ..survival_curves import breslow_baseline, cox_survival, ipcw_ibs, time_grid
from .losses import cox_ph_loss, cox_survival_curve, survival_curve_distill

torch.set_num_threads(1)


def _t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def _eta(model, X, M):
    """Cox linear predictor (mixed) for arrays X, M."""
    with torch.no_grad():
        out = model(_t(X), _t(M))
    return out["mixed"][:, 0].cpu().numpy()


def train_single(Xtr, ttr, etr, Xva, tva, eva, cfg) -> HierarchicalSubtypeSurvival:
    """Trunk + agnostic Cox head (n_subtypes=0). Used for pooled and per-subtype
    independent baselines depending on which patients are passed in."""
    return _train(Xtr, np.zeros((len(Xtr), 0), "float32"), ttr, etr,
                  Xva, np.zeros((len(Xva), 0), "float32"), tva, eva, cfg, distill=False)


def train_hss(Xtr, Mtr, ttr, etr, Xva, Mva, tva, eva, cfg) -> HierarchicalSubtypeSurvival:
    return _train(Xtr, Mtr, ttr, etr, Xva, Mva, tva, eva, cfg, distill=True)


def _train(Xtr, Mtr, ttr, etr, Xva, Mva, tva, eva, cfg, distill: bool):
    tr = cfg.get("training", {})
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed); np.random.seed(seed)
    # Fair early-stopping for small subtypes: if the supplied val has <2 events,
    # carve an internal 20% holdout from train (keeps the baseline strong).
    Xtr, Mtr, ttr, etr = map(np.asarray, (Xtr, Mtr, ttr, etr))
    if np.asarray(eva).sum() < 2 and len(Xtr) >= 10:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Xtr)); k = max(2, int(0.2 * len(Xtr)))
        es, fit = perm[:k], perm[k:]
        Xva, Mva, tva, eva = Xtr[es], Mtr[es], ttr[es], etr[es]
        Xtr, Mtr, ttr, etr = Xtr[fit], Mtr[fit], ttr[fit], etr[fit]
    n_sub = Mtr.shape[1]
    lam = float(tr.get("distill_weight", 1.0))
    start = int(tr.get("distill_start_epoch", 20))
    epochs, patience = int(tr.get("epochs", 200)), int(tr.get("patience", 30))
    use_ema = bool(tr.get("ema_teacher", False))
    ema_decay = float(tr.get("ema_decay", 0.98))

    model = HierarchicalSubtypeSurvival(Xtr.shape[1], n_sub, head_type="cox",
                                        hidden=int(tr.get("hidden_dim", 64)),
                                        dropout=float(tr.get("dropout", 0.1)))
    opt = torch.optim.Adam(model.parameters(), lr=float(tr.get("lr", 2e-3)),
                           weight_decay=float(tr.get("weight_decay", 1e-4)))
    teacher = make_ema_teacher(model) if (distill and use_ema) else None

    xt, Mt, tt, et = _t(Xtr), _t(Mtr), _t(ttr), _t(etr)
    grid = time_grid(ttr, etr, ttr)
    gt = _t(grid) if grid is not None else None

    best, best_ep, best_state = -np.inf, -1, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(xt, Mt)
        eta_mix, eta_ag = out["mixed"][:, 0], out["agnostic"][:, 0]
        loss = cox_ph_loss(eta_mix, tt, et, ties="efron") + cox_ph_loss(eta_ag, tt, et, ties="efron")
        for s in range(n_sub):
            mask = Mt[:, s] > 0.5
            if mask.sum() >= 3 and et[mask].sum() >= 1:
                loss = loss + cox_ph_loss(out["subs"][mask, s, 0], tt[mask], et[mask], ties="efron")
        if distill and n_sub > 0 and ep >= start and gt is not None:
            # cross-head survival-CURVE distillation: subtype head -> agnostic teacher
            with torch.no_grad():
                ema_ag = teacher(xt, Mt)["agnostic"][:, 0] if teacher is not None else eta_ag
                H = _t(breslow_baseline(ttr, etr, ema_ag.cpu().numpy(), grid))
            dl = torch.zeros(())
            for s in range(n_sub):
                mask = Mt[:, s] > 0.5
                if mask.sum() < 3:
                    continue
                Ss = cox_survival_curve(out["subs"][mask, s, 0], H)
                S0 = cox_survival_curve(eta_ag[mask].detach(), H)
                dl = dl + survival_curve_distill(Ss, S0)
            loss = loss + lam * dl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if teacher is not None:
            update_ema_agnostic(model, teacher, ema_decay)

        # selection on VAL only (never test)
        if eva.sum() >= 2:
            model.eval()
            cv = fast_c_index(tva, eva, _eta(model, Xva, Mva))
            if np.isfinite(cv) and cv > best:
                best, best_ep = cv, ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif ep - best_ep >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def per_subtype_ibs(model, X, M, t, e, split, subtype_cols, cfg,
                    independent_fn=None) -> list[dict]:
    """Per-subtype IPCW-IBS + C-index on the held-out fold. If independent_fn is
    given, also trains a per-subtype independent baseline for the same subtype."""
    tr = split == "train"
    te = split == "test"
    rows = []
    for i, s in enumerate(subtype_cols):
        mem = M[:, i] > 0.5
        s_tr, s_te = tr & mem, te & mem
        n_te, ev_te = int(s_te.sum()), int(e[s_te].sum())
        if n_te < 3 or ev_te < 1 or s_tr.sum() < 8:
            rows.append({"subtype": s, "test_n": n_te, "test_ev": ev_te, "status": "too_sparse"})
            continue
        grid = time_grid(t[s_tr], e[s_tr], t[s_te])
        if grid is None:
            rows.append({"subtype": s, "test_n": n_te, "status": "no_grid"})
            continue
        eta_tr = _eta(model, X[s_tr], M[s_tr]); eta_te = _eta(model, X[s_te], M[s_te])
        H = breslow_baseline(t[s_tr], e[s_tr], eta_tr, grid)
        ibs_hss = ipcw_ibs(t[s_tr], e[s_tr], t[s_te], e[s_te], cox_survival(eta_te, H), grid)
        row = {"subtype": s, "test_n": n_te, "test_ev": ev_te,
               "ibs_hss": round(ibs_hss, 4), "c_hss": round(fast_c_index(t[s_te], e[s_te], eta_te), 3),
               "status": "ok"}
        if independent_fn is not None:
            ind = independent_fn(i)
            eta_tr_i = _eta(ind, X[s_tr], np.zeros((int(s_tr.sum()), 0), "float32"))
            eta_te_i = _eta(ind, X[s_te], np.zeros((n_te, 0), "float32"))
            Hi = breslow_baseline(t[s_tr], e[s_tr], eta_tr_i, grid)
            ibs_ind = ipcw_ibs(t[s_tr], e[s_tr], t[s_te], e[s_te], cox_survival(eta_te_i, Hi), grid)
            row["ibs_independent"] = round(ibs_ind, 4)
            row["ibs_delta_hss_minus_ind"] = round(ibs_hss - ibs_ind, 4)
            row["hss_better"] = bool(ibs_hss <= ibs_ind)
        rows.append(row)
    return rows
