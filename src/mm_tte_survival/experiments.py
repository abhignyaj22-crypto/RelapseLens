from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from .config import ensure_outdir
from .data import prepare_dataset
from .metrics import harrell_c_index, bootstrap_ci
from .training.trainer_legacy import fit_neural_survival, subtype_event_rate_baseline


def run_experiments(cfg: dict) -> dict:
    outdir = ensure_outdir(cfg["paths"].get("outdir", "outputs/run"))
    bundle = prepare_dataset(cfg)
    rows = []
    predictions = pd.DataFrame({"patient_id": bundle.ids_test, "time": bundle.t_test, "event": bundle.e_test})
    histories = {}
    for name in cfg.get("experiments", {}).get("models", ["cox", "opsd_cox", "aft", "opsd_aft", "fht", "opsd_fht"]):
        if name == "subtype_event_rate":
            res = subtype_event_rate_baseline(bundle)
        else:
            res = fit_neural_survival(bundle, name, cfg)
        lo, hi = bootstrap_ci(bundle.t_test, bundle.e_test, res.risk_test, n_boot=int(cfg.get("experiments", {}).get("bootstrap", 80)), seed=int(cfg.get("seed", 42)))
        rows.append({
            "model": name,
            "val_cindex": res.cindex_val,
            "test_cindex": res.cindex_test,
            "test_cindex_ci_low": lo,
            "test_cindex_ci_high": hi,
            "risk_event_proxy_at_horizon": res.risk_event_proxy_at_horizon,
            "n_test": len(bundle.t_test),
            "events_test": int(np.sum(bundle.e_test)),
        })
        predictions[f"risk_{name}"] = res.risk_test
        histories[name] = res.history
        if res.model is not None:
            torch.save({"model_type": name, "state_dict": res.model.state_dict(), "feature_names": bundle.feature_names}, outdir / f"model_{name}.pt")
    leaderboard = pd.DataFrame(rows).sort_values("test_cindex", ascending=False)
    leaderboard.to_csv(outdir / "leaderboard.csv", index=False)
    predictions.to_csv(outdir / "test_predictions.csv", index=False)
    (outdir / "training_histories.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    subtype = subtype_metrics(bundle, predictions, cfg)
    subtype.to_csv(outdir / "per_subtype_metrics.csv", index=False)
    manifest = {
        "n_train": int(len(bundle.t_train)), "events_train": int(np.sum(bundle.e_train)),
        "n_val": int(len(bundle.t_val)), "events_val": int(np.sum(bundle.e_val)),
        "n_test": int(len(bundle.t_test)), "events_test": int(np.sum(bundle.e_test)),
        "n_features": int(len(bundle.feature_names)), "subtype_cols": bundle.subtype_cols,
        "best_model": leaderboard.iloc[0].to_dict() if len(leaderboard) else None,
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"outdir": str(outdir), "leaderboard": leaderboard, "subtype_metrics": subtype, "manifest": manifest}


def subtype_metrics(bundle, predictions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    min_n = int(cfg.get("experiments", {}).get("min_subtype_n", 15))
    min_e = int(cfg.get("experiments", {}).get("min_subtype_events", 5))
    merged = bundle.merged[bundle.merged["split"].eq("test")].copy()
    risk_cols = [c for c in predictions.columns if c.startswith("risk_")]
    merged = merged.merge(predictions[["patient_id"] + risk_cols], on="patient_id", how="inner")
    rows = []
    for subtype in bundle.subtype_cols:
        m = merged[subtype].fillna(0) > 0
        n = int(m.sum())
        events = int(merged.loc[m, "event"].sum()) if "event" in merged else 0
        if n < min_n or events < min_e:
            rows.append({"subtype": subtype, "n": n, "events": events, "status": "too_sparse"})
            continue
        for r in risk_cols:
            rows.append({
                "subtype": subtype, "model": r.replace("risk_", ""), "n": n, "events": events,
                "cindex": harrell_c_index(merged.loc[m, "time_months"], merged.loc[m, "event"], merged.loc[m, r]),
                "status": "ok",
            })
    return pd.DataFrame(rows)
