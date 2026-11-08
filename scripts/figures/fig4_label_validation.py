#!/usr/bin/env python
"""Figure 4 — label-validation panel ("labels trustworthy enough without FISH").

Three panels, all open-data:
  A. External validation — real FISH (GSE6477: del13, hyperdiploid) + expression-
     cluster concordance (GSE19784: translocations). AUC per subtype; FISH vs cluster
     distinguished. t(14;16) visibly fails.
  B. Internal cross-modality concordance — CoMMpass CNV calls vs orthogonal RNA dosage.
  C. Label-noise robustness — fraction of draws where pooled penalised Cox stays
     no-worse-than subtype-specific after flipping labels at published discordance rates.

PLOT-ONLY. Reads outputs/validation/{external_geo,internal_concordance}.csv and
outputs/validation/label_noise/label_noise_detail.csv. NOT FISH where stated.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import COLORS, save, plt

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "outputs" / "validation"
FLIP_RATE = {"del17p": 0.11, "amp1q": 0.29, "del13q": 0.30, "del1p": 0.30, "hyperdiploid": 0.10}


def _panel_external(ax):
    df = pd.read_csv(V / "external_geo.csv")
    df = df.dropna(subset=["auc"]).copy()
    df["kind"] = np.where(df["is_fish"], "real FISH (GSE6477)", "expr cluster (GSE19784)")
    df = df.sort_values(["is_fish", "auc"], ascending=[False, True])
    colors = [COLORS["confirmed"] if f else COLORS["caution"] for f in df["is_fish"]]
    ys = range(len(df))
    ax.barh(list(ys), df["auc"], color=colors, alpha=0.85)
    ax.axvline(0.5, color="#999", ls=":", lw=0.8)
    for y, (_, r) in zip(ys, df.iterrows()):
        ax.text(r["auc"] + 0.01, y, f"{r['auc']:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(list(ys)); ax.set_yticklabels(df["subtype"], fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("AUC vs gold standard", fontsize=8.5)
    ax.set_title("A. External validation", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS["confirmed"], label="real FISH"),
                       Patch(color=COLORS["caution"], label="expr cluster (NOT FISH)")],
              fontsize=6.8, loc="center right", framealpha=0.9)


def _panel_internal(ax):
    df = pd.read_csv(V / "internal_concordance.csv").dropna(subset=["auc"]).sort_values("auc")
    ys = range(len(df))
    ax.barh(list(ys), df["auc"], color=COLORS["omics"], alpha=0.85)
    ax.axvline(0.5, color="#999", ls=":", lw=0.8)
    for y, (_, r) in zip(ys, df.iterrows()):
        ax.text(r["auc"] + 0.01, y, f"{r['auc']:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(list(ys)); ax.set_yticklabels(df["subtype"], fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("AUC: CNV call vs orthogonal RNA", fontsize=8.5)
    ax.set_title("B. Internal concordance (NOT FISH)", fontsize=9)


def _panel_noise(ax):
    det = pd.read_csv(V / "label_noise" / "label_noise_detail.csv")
    flip = det[det.kind == "flipped"]
    fracs = (flip.assign(not_worse=flip["pooled_minus_independent"] <= 0.01)
             .groupby("subtype")["not_worse"].mean())
    fracs = fracs.sort_values()
    ys = range(len(fracs))
    ax.barh(list(ys), fracs.values, color=COLORS["clinical"], alpha=0.85)
    ax.axvline(0.8, color=COLORS["blocked"], ls="--", lw=1, label="robust threshold 0.80")
    for y, v in zip(ys, fracs.values):
        ax.text(min(v + 0.01, 0.98), y, f"{v:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{s} (flip {int(FLIP_RATE.get(s,0)*100)}%)" for s in fracs.index], fontsize=7.5)
    ax.set_xlim(0, 1.05); ax.set_xlabel("frac. draws pooled Cox not-worse", fontsize=8.5)
    ax.set_title("C. Label-noise robustness", fontsize=9)
    ax.legend(fontsize=6.8, loc="lower left", framealpha=0.9)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    _panel_external(axes[0]); _panel_internal(axes[1]); _panel_noise(axes[2])
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Sequencing-inferred subtype labels are trustworthy enough without CoMMpass FISH "
                 "(del13 best-supported; del1p / t(14;16) most uncertain)", fontsize=10, y=1.04)
    fig.text(0.01, -0.04, "No FISH exists for the open CoMMpass cohort (MMRF seqFISH is "
             "controlled-access). OS technical validation only.", fontsize=6.8, color="#555555")
    fig.tight_layout()
    save(fig, "fig4_label_validation")


if __name__ == "__main__":
    main()
