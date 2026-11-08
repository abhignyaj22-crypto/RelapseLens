#!/usr/bin/env python
"""Supplement Figure S3 — mmSYGNAL 141-program activity QC (NOT a validation cohort).

The upstream example is N=3 and is used only as a format/distribution sanity note,
never as validation. Panels:
  A. {-1,0,+1} program-activity distribution (ours, N=787)
  B. histogram of per-program mean activity
  C. mmSYGNAL model-type selection counts (agnostic/amp1q/del13q/del1p/t(4;14))
  D. mmSYGNAL risk-class counts (low/high/extreme)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import COLORS, save, plt

ROOT = Path(__file__).resolve().parents[2]
PROG = ROOT / "data" / "real" / "mmsygnal_program_activity_0_140.csv"
SCORES = ROOT / "outputs" / "benchmarks" / "mmSYGNAL" / "mmsygnal_scores.csv"
CYTO = ROOT / "data" / "real" / "cytogenetics.csv"


def main():
    prog = pd.read_csv(PROG)
    pcols = [str(i) for i in range(141)]
    v = prog[pcols].values.astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # A. {-1,0,1} distribution
    ax = axes[0, 0]
    fr = [np.mean(v == k) for k in (-1, 0, 1)]
    ax.bar(["-1", "0", "+1"], fr, color=COLORS["omics"], edgecolor="black", lw=0.4)
    for i, f in enumerate(fr):
        ax.text(i, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("fraction"); ax.set_xlabel("program activity value")
    ax.set_title(f"A. Activity distribution (N=787, mean={v.mean():.2f})")

    # B. histogram of per-program mean activity
    ax = axes[0, 1]
    ax.hist(v.mean(axis=0), bins=20, color=COLORS["model"], edgecolor="black", lw=0.3)
    ax.set_xlabel("per-program mean activity (141 programs)"); ax.set_ylabel("programs")
    ax.set_title("B. Per-program mean activity")

    # C. model-type selection counts
    ax = axes[1, 0]
    if SCORES.exists() and CYTO.exists():
        s = pd.read_csv(SCORES); s["patient_id"] = s.patient_id.astype(str)
        cy = pd.read_csv(CYTO); cy["patient_id"] = cy.patient_id.astype(str)
        m = s.merge(cy, on="patient_id", how="left")
        # reproduce grade-based selection counts (A: t_4_14 > B: amp1q/del13q/del1p > C: agnostic)
        def col(name):
            return (pd.to_numeric(m[name], errors="coerce").fillna(0).to_numpy() if name in m.columns
                    else np.zeros(len(m)))
        sel = np.select([col("t_4_14") == 1, col("amp1q") == 1, col("del13q") == 1, col("del1p") == 1],
                        ["t(4;14)", "amp1q", "del13q", "del1p"], default="agnostic")
        vc = pd.Series(sel).value_counts().reindex(["t(4;14)", "amp1q", "del13q", "del1p", "agnostic"]).fillna(0)
        ax.bar(vc.index, vc.values, color=COLORS["cytogenetics"], edgecolor="black", lw=0.4)
        for i, c in enumerate(vc.values):
            ax.text(i, c + 3, int(c), ha="center", fontsize=8)
        ax.set_ylabel("patients"); ax.set_title("C. mmSYGNAL model-type selection")
        ax.tick_params(axis="x", labelsize=8)

    # D. risk-class counts
    ax = axes[1, 1]
    if SCORES.exists():
        s = pd.read_csv(SCORES)
        rc = s["mmsygnal_risk_class"].value_counts().reindex(["low", "high", "extreme"]).fillna(0)
        ax.bar(rc.index, rc.values, color=[COLORS["confirmed"], COLORS["caution"], COLORS["blocked"]],
               edgecolor="black", lw=0.4)
        for i, c in enumerate(rc.values):
            ax.text(i, c + 3, int(c), ha="center", fontsize=9)
        ax.set_ylabel("patients"); ax.set_title("D. mmSYGNAL risk-class (low/high/extreme)")

    fig.suptitle("S3. mmSYGNAL 141-program activity QC — official miner3 (method-reproduced, "
                 "NOT a validation cohort)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "sfig3_mmsygnal_context")


if __name__ == "__main__":
    main()
