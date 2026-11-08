#!/usr/bin/env python
"""Main Figure 5 — molecular residual-risk reclassification.

Panels (all from outputs CSVs):
  A. clinical-risk tertile x molecular-residual tertile heatmap (counts)
  B. reclassified up/down counts
  C. KM: clinical-low/molecular-high vs clinical-low/molecular-low
  D. hazard ratios for reclassification groups
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import COLORS, save, plt
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "experiment0_open_gdc_os"


def _tertiles(x):
    return pd.qcut(pd.Series(x).rank(method="first"), 3, labels=["low", "mid", "high"]).astype(object).values


def main():
    dec = pd.read_csv(D / "residual_risk_decomposition.csv")
    out = pd.read_csv(D / "mmrf_reclassification_outcomes.csv")
    tcol = "time_months" if "time_months" in dec.columns else dec.columns[2]
    ecol = "event"
    dec["clin_t"] = _tertiles(dec["clinical_risk"].values)
    dec["tot_t"] = _tertiles(dec["total_risk"].values)
    dec["res_t"] = _tertiles(dec["molecular_residual_risk"].values)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # A. heatmap clinical tertile x residual tertile
    order = ["low", "mid", "high"]
    ct = pd.crosstab(pd.Categorical(dec["clin_t"], order), pd.Categorical(dec["res_t"], order))
    ax = axes[0, 0]
    im = ax.imshow(ct.values, cmap="Greens")
    ax.set_xticks(range(3)); ax.set_xticklabels(order); ax.set_yticks(range(3)); ax.set_yticklabels(order)
    ax.set_xlabel("molecular-residual tertile"); ax.set_ylabel("clinical-risk tertile")
    ax.set_title("A. Clinical vs molecular-residual risk (counts)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, ct.values[i, j], ha="center", va="center", fontsize=10)

    # B. reclassified up/down (total tertile vs clinical tertile)
    rk = {"low": 0, "mid": 1, "high": 2}
    moved = dec[dec.clin_t != dec.tot_t]
    up = int((moved.apply(lambda r: rk[r.tot_t] > rk[r.clin_t], axis=1)).sum())
    down = int((moved.apply(lambda r: rk[r.tot_t] < rk[r.clin_t], axis=1)).sum())
    same = int(len(dec) - len(moved))
    ax = axes[0, 1]
    ax.bar(["unchanged", "reclassified up", "reclassified down"], [same, up, down],
           color=[COLORS["model"], COLORS["blocked"], COLORS["confirmed"]], edgecolor="black", lw=0.4)
    ax.set_ylabel("patients"); ax.set_title(f"B. Reclassification by omics (N={len(dec)})")
    for i, v in enumerate([same, up, down]):
        ax.text(i, v + 3, str(v), ha="center", fontsize=9)

    # C. KM clinical-low molecular-high vs molecular-low
    clin_bin = np.where(pd.Series(dec.clinical_risk.values).rank(pct=True) > 0.5, "high", "low")
    mol_bin = np.where(pd.Series(dec.molecular_residual_risk.values).rank(pct=True) > 0.5, "high", "low")
    ax = axes[1, 0]
    t, e = dec[tcol].values, dec[ecol].values
    for lab, mask, col in [("clinical-low / molecular-low", (clin_bin == "low") & (mol_bin == "low"), COLORS["confirmed"]),
                           ("clinical-low / molecular-high", (clin_bin == "low") & (mol_bin == "high"), COLORS["caution"])]:
        if mask.sum() >= 5:
            KaplanMeierFitter().fit(t[mask], e[mask], label=f"{lab} (n={int(mask.sum())})").plot_survival_function(ax=ax, ci_show=False, color=col)
    ax.set_title("C. OS within clinical-low stratum"); ax.set_xlabel("months"); ax.set_ylabel("survival")

    # D. WITHIN-STRATUM hazard ratios (molecular-high vs molecular-low) — corrects
    #    the misleading "HR vs rest" comparison.
    ax = axes[1, 1]
    ws = pd.read_csv(D / "reclassification_within_stratum_hr.csv")
    ws = ws[ws["analysis"] == "within_stratum"].copy()
    ax.axvline(1.0, color="gray", ls=":")
    ax.errorbar(ws["HR"], range(len(ws)),
                xerr=[ws["HR"] - ws["ci_low"], ws["ci_high"] - ws["HR"]],
                fmt="o", color=COLORS["model"], capsize=4, ms=7)
    ax.set_yticks(range(len(ws)))
    ax.set_yticklabels([f"{s}: mol-high vs low" for s in ws["stratum"]], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("hazard ratio (within stratum)")
    ax.set_title("D. Within-stratum HRs (molecular high vs low)")
    for i, (_, r) in enumerate(ws.iterrows()):
        ax.text(r["ci_high"] * 1.05, i, f"HR {r['HR']:.2f}, p={r['p']:.1e}", va="center", fontsize=7.5)

    fig.suptitle("Molecular residual-risk reclassification (open-GDC OS, hypothesis-generating)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "sfig4_reclassification")


if __name__ == "__main__":
    main()
