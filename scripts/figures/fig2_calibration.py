#!/usr/bin/env python
"""Figure 2 — calibration from REAL predicted survival (not sigmoid-of-risk).

The old pipeline scored calibration off a sigmoid of a z-scored risk; the corrected
pipeline uses proper predicted survival S(tau) and reports IPCW-IBS + a reliability
curve. Two panels:
  A. Reliability — observed vs predicted 24-month event probability by decile, with
     calibration slope / in-the-large annotated (ideal = diagonal).
  B. IPCW integrated Brier score per leak-proof model class (lower = better).

PLOT-ONLY. Reads the canonical run's calibration_by_decile.csv + calibration_metrics.csv
and the leak-proof rebaseline leakproof_ipcw_ibs.csv. Endpoint = OS technical validation.
"""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import color_for, COLORS, save, plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs" / "experiment0_open_gdc_os"
REB = ROOT / "outputs" / "rebaseline_open_gdc_os"
SHOW = ["clinical", "clinical+omics", "clinical+cytogenetics+omics"]


def _panel_reliability(ax):
    dec = pd.read_csv(RUN / "calibration_by_decile.csv")
    met = pd.read_csv(RUN / "calibration_metrics.csv").set_index("model")
    ax.plot([0, 0.5], [0, 0.5], color="#999", ls="--", lw=1, label="ideal")
    for m in SHOW:
        d = dec[dec.model == m].sort_values("pred_event_prob")
        if d.empty:
            continue
        slope = met.loc[m, "calibration_slope"] if m in met.index else float("nan")
        ax.plot(d["pred_event_prob"], d["observed_event_prob"], "o-", color=color_for(m),
                ms=4, lw=1.5, label=f"{m} (slope {slope:.2f})")
    ax.set_xlabel("predicted 24-mo event prob", fontsize=8.5)
    ax.set_ylabel("observed 24-mo event prob", fontsize=8.5)
    ax.set_title("A. Reliability — real S(τ), not sigmoid-of-risk", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)


def _panel_ibs(ax):
    ibs = pd.read_csv(REB / "leakproof_ipcw_ibs.csv")
    ibs = ibs[ibs.k == 16].copy()
    keep = {"clinical": "clinical", "clinical+omics_infold": "clinical+omics",
            "clinical+cyto+omics_infold": "clinical+cyto+omics"}
    ibs = ibs[ibs.feature_set.isin(keep)].copy()
    ibs["lab"] = ibs.feature_set.map(keep)
    ibs = ibs.sort_values("mean_ibs", ascending=False)
    ys = range(len(ibs))
    ax.barh(list(ys), ibs["mean_ibs"], color=[color_for(x) for x in ibs["lab"]], alpha=0.85)
    for y, (_, r) in zip(ys, ibs.iterrows()):
        ax.text(r["mean_ibs"] + 0.001, y, f"{r['mean_ibs']:.3f}", va="center", fontsize=8)
    ax.set_yticks(list(ys)); ax.set_yticklabels(ibs["lab"], fontsize=8)
    ax.set_xlabel("IPCW integrated Brier score (k=16, in-fold)", fontsize=8.5)
    ax.set_title("B. IPCW-IBS by model class (lower=better)", fontsize=9)
    ax.set_xlim(0, max(ibs["mean_ibs"]) * 1.25)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    _panel_reliability(axes[0]); _panel_ibs(axes[1])
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.01, -0.03, "Calibration computed from real predicted survival probabilities "
             "(corrected losses; see SFig loss-correctness). OS technical validation only.",
             fontsize=6.8, color="#555555")
    fig.tight_layout()
    save(fig, "fig2_calibration")


if __name__ == "__main__":
    main()
