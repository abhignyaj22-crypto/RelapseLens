#!/usr/bin/env python
"""Figure 1 (headline) — the ceiling forest plot.

Harrell C-index with 95% CI for the leak-proof model classes on OS, with the
honest open-data SOTA band (~0.62-0.65) shaded. The central result: adding omics
lifts discrimination modestly but everything lands in/near the same band — and the
neural and subtype-conditioned models do NOT exceed it (Stage D / Direction-2 null).

PLOT-ONLY (Fragility 1): every value is read from the canonical run's
repeated_split_leaderboard.csv (50 patient-disjoint splits); this script fits
nothing. Endpoint = OS technical validation; cytogenetics are sequencing-inferred,
NOT FISH.
"""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import color_for, COLORS, save, plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs" / "experiment0_open_gdc_os"
SOTA_LO, SOTA_HI = 0.62, 0.65   # honest open-data MM-OS ceiling (gep70/sky92-class)

LABELS = {
    "clinical": "Clinical",
    "clinical+cytogenetics": "Clinical + cyto (seq-inferred)",
    "clinical+omics": "Clinical + omics (in-fold PCA)",
    "clinical+cytogenetics+omics": "Clinical + cyto + omics",
}


def main():
    df = pd.read_csv(RUN / "repeated_split_leaderboard.csv").set_index("feature_set")
    df = df.loc[[k for k in LABELS if k in df.index]]

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.axvspan(SOTA_LO, SOTA_HI, color="#bdbdbd", alpha=0.35, zorder=0,
               label=f"honest open-data ceiling (~{SOTA_LO:.2f}–{SOTA_HI:.2f})")
    ax.axvline(0.5, color="#999999", lw=0.8, ls=":", zorder=0)
    ys = range(len(df))
    for y, (fs, r) in zip(ys, df.iterrows()):
        c = color_for(fs)
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=c, lw=2.2, zorder=2)
        ax.plot(r["mean_cindex"], y, "o", color=c, ms=7, zorder=3)
        ax.text(r["ci_high"] + 0.004, y,
                f"{r['mean_cindex']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
                va="center", ha="left", fontsize=8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([LABELS[fs] for fs in df.index], fontsize=9)
    ax.set_xlim(0.48, 0.90)
    ax.set_xlabel("Harrell C-index (OS) — 50 patient-disjoint splits, mean ± 95% CI", fontsize=9)
    ax.set_title("Discrimination ceiling on open MMRF-CoMMpass OS\n"
                 "neural & subtype-conditioned models do not exceed this band (Stage D / Direction-2 null)",
                 fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.26, top=0.84)
    fig.text(0.02, 0.02, "OS technical validation only — not PFS/relapse. Cytogenetics are "
             "sequencing-inferred (NOT FISH).", fontsize=6.8, color="#555555")
    save(fig, "fig1_ceiling_forest")


if __name__ == "__main__":
    main()
