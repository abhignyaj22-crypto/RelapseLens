#!/usr/bin/env python
"""Supplement Figure S4 — repeated-split stability (distributions, not point bars).

A. box+strip of OS C-index across the stratified splits per feature set
B. paired ΔC distribution vs clinical with the zero line emphasised
Emphasises: stable positive direction but ΔC CI crosses 0 (NOT confirmatory).

PLOT-ONLY. All per-split values are computed by
scripts/analysis/repeated_split_detail.py and read from
outputs/.../repeated_split_*.csv. This script fits no models (Fragility 1:
figure scripts must not be the source of a manuscript number).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import color_for, COLORS, save, plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "experiment0_open_gdc_os"
PER_MODEL = OUT / "repeated_split_per_model.csv"
PAIRED = OUT / "repeated_split_paired_delta.csv"


def main():
    if not PER_MODEL.exists() or not PAIRED.exists():
        sys.exit(f"missing {PER_MODEL.name}/{PAIRED.name}; run "
                 "`python scripts/analysis/repeated_split_detail.py` first (or `make analysis`).")
    pm = pd.read_csv(PER_MODEL)
    pd_delta = pd.read_csv(PAIRED)

    per = {k: pm.loc[pm.feature_set == k, "test_cindex"].tolist()
           for k in pm.feature_set.unique()}
    order = sorted(per, key=lambda k: np.mean(per[k]))
    d_om = pd_delta.loc[pd_delta.comparison == "+omics vs clinical", "delta_cindex"].values
    d_full = pd_delta.loc[pd_delta.comparison == "+cyto+omics vs clinical", "delta_cindex"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    data = [per[k] for k in order]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6, showfliers=False)
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(color_for(k)); patch.set_alpha(0.6)
    for i, k in enumerate(order, 1):
        ax.scatter(per[k], np.random.default_rng(0).normal(i, 0.05, len(per[k])), s=6, color="black", alpha=0.4)
    ax.set_yticklabels(order, fontsize=9); ax.set_xlabel(f"OS C-index across {len(data[0])} splits")
    ax.set_title("A. Per-split C-index distribution")

    ax = axes[1]
    parts = ax.violinplot([d_om, d_full], vert=False, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(COLORS["omics"]); pc.set_alpha(0.5)
    ax.axvline(0, color=COLORS["blocked"], ls="--", lw=1.2, label="no improvement")
    ax.set_yticks([1, 2]); ax.set_yticklabels(["+omics vs clinical", "+cyto+omics vs clinical"], fontsize=9)
    ax.set_xlabel("paired ΔC-index")
    f_om, f_full = np.mean(d_om > 0), np.mean(d_full > 0)
    ax.set_title("B. Paired ΔC distribution")
    ax.text(0.5, -0.22, f"improved in {f_om*100:.0f}% / {f_full*100:.0f}% of splits — "
            "stable positive direction, but ΔC CI crosses 0 (NOT confirmatory).",
            transform=ax.transAxes, ha="center", fontsize=8, color=COLORS["caution"])
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("S4. Repeated stratified-split stability (open-GDC OS)", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save(fig, "sfig2_repeated_split_stability")


if __name__ == "__main__":
    main()
