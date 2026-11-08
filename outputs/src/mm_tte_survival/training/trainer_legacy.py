from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

# Small survival batches plus repeated model fits can be slower/unstable with
# excessive OpenMP threading in constrained environments. Keep deterministic,
# low-overhead CPU execution by default.
torch.set_num_threads(1)

from .losses import cox_ph_loss, lognormal_aft_loss, first_hitting_time_loss, risk_from_output, soft_distillation_loss
from ..metrics import harrell_c_index, risk_event_proxy_at_horizon
from ..models import SurvivalMLP, make_ema_teacher, update_ema_teacher


@dataclass
class FitResult:
    model_type: str
    model: nn.Module | None
    risk_test: np.ndarray
    risk_val: np.ndarray
    cindex_val: float
    cindex_test: float
    risk_event_proxy_at_horizon: float
    history: list[dict]


def _supervised_loss(model_type: str, output: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    base = model_type.replace("opsd_", "")
    if base == "cox":
        return cox_ph_loss(output.reshape(-1), time, event)
    if base == "aft":
        return lognormal_aft_loss(output, time, event)
    if base == "fht":
        return first_hitting_time_loss(output, time, event)
    raise ValueError(f"Unknown model_type: {model_type}")


def fit_neural_survival(bundle, model_type: str, cfg: dict) -> FitResult:
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = cfg.get("training", {})
    base_type = model_type.replace("opsd_", "")
    use_distill = model_type.startswith("opsd_")
    model = SurvivalMLP(bundle.x_train.shape[1], base_type, int(tr.get("hidden_dim", 64)), float(tr.get("dropout", 0.1)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr.get("lr", 2e-3)), weight_decay=float(tr.get("weight_decay", 1e-4)))
    teacher = make_ema_teacher(model) if use_distill else None

    x = torch.tensor(bundle.x_train, dtype=torch.float32)
    t = torch.tensor(bundle.t_train, dtype=torch.float32)
    e = torch.tensor(bundle.e_train, dtype=torch.float32)
    xv = torch.tensor(bundle.x_val, dtype=torch.float32)
    xt = torch.tensor(bundle.x_test, dtype=torch.float32)

    best_state = None
    best_val = -np.inf
    best_epoch = -1
    patience = int(tr.get("patience", 25))
    history = []
    for epoch in range(int(tr.get("epochs", 100))):
        model.train()
        opt.zero_grad()
        out = model(x)
        supervised = _supervised_loss(model_type, out, t, e)
        loss = supervised
        distill = torch.tensor(0.0, device=x.device)
        if use_distill and teacher is not None:
            # OPSD should regularize a competent on-policy model, not dominate the
            # early survival objective. A warm-up + ramp prevents the EMA teacher
            # from anchoring a near-random early ranking, which is a common cause
            # of C-index collapse on small censored datasets.
            start = int(tr.get("distill_start_epoch", max(5, int(tr.get("epochs", 100)) // 5)))
            ramp_epochs = max(1, int(tr.get("distill_ramp_epochs", 10)))
            if epoch >= start:
                with torch.no_grad():
                    teach_out = teacher(x)
                    teach_risk = risk_from_output(base_type, teach_out)
                stud_risk = risk_from_output(base_type, out)
                if torch.std(teach_risk, unbiased=False) > 1e-6:
                    ramp = min(1.0, float(epoch - start + 1) / float(ramp_epochs))
                    w = float(tr.get("distill_weight", 0.05)) * ramp
                    distill = soft_distillation_loss(stud_risk, teach_risk)  # temperature was a no-op; removed
                    loss = supervised + w * distill
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if use_distill and teacher is not None:
            update_ema_teacher(model, teacher, float(tr.get("ema_decay", 0.97)))

        # selection + logging on VALIDATION ONLY. The test set is scored exactly
        # once, after the model is locked (below) — no per-epoch test peeking.
        model.eval()
        with torch.no_grad():
            rv = risk_from_output(base_type, model(xv)).cpu().numpy()
        cv = harrell_c_index(bundle.t_val, bundle.e_val, rv)
        history.append({
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "supervised_loss": float(supervised.detach().cpu()),
            "distill_loss": float(distill.detach().cpu()),
            "val_cindex": float(cv),
        })
        if np.isfinite(cv) and cv > best_val:
            best_val = cv
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rv = risk_from_output(base_type, model(xv)).cpu().numpy()
        rt = risk_from_output(base_type, model(xt)).cpu().numpy()
    return FitResult(
        model_type=model_type,
        model=model,
        risk_test=rt,
        risk_val=rv,
        cindex_val=harrell_c_index(bundle.t_val, bundle.e_val, rv),
        cindex_test=harrell_c_index(bundle.t_test, bundle.e_test, rt),
        risk_event_proxy_at_horizon=risk_event_proxy_at_horizon(bundle.t_test, bundle.e_test, rt),
        history=history,
    )


def subtype_event_rate_baseline(bundle) -> FitResult:
    # Estimate subtype-level event pressure on train; fall back to global event rate.
    merged = bundle.merged.copy()
    train = merged[merged["split"].eq("train")]
    global_rate = float(train["event"].mean()) if "event" in train else 0.5
    subtype_cols = bundle.subtype_cols
    weights = {}
    for c in subtype_cols:
        m = train[c].fillna(0) > 0
        weights[c] = float(train.loc[m, "event"].mean()) if m.sum() >= 3 else global_rate
    def score(df):
        vals = np.repeat(global_rate, len(df)).astype(float)
        for c in subtype_cols:
            vals += df[c].fillna(0).to_numpy(dtype=float) * (weights[c] - global_rate)
        return vals
    val_df = merged[merged["split"].eq("val")]
    test_df = merged[merged["split"].eq("test")]
    rv = score(val_df)
    rt = score(test_df)
    return FitResult(
        model_type="subtype_event_rate", model=None, risk_val=rv, risk_test=rt,
        cindex_val=harrell_c_index(bundle.t_val, bundle.e_val, rv),
        cindex_test=harrell_c_index(bundle.t_test, bundle.e_test, rt),
        risk_event_proxy_at_horizon=risk_event_proxy_at_horizon(bundle.t_test, bundle.e_test, rt), history=[])
