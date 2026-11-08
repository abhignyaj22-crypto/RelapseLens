#!/usr/bin/env python
"""Supplementary Figure — Stage-D subtype calibration/IBS negative-control analysis.

Small-subtype calibration improvement (independent − HSS IBS) under the observed
CNV-derived labels vs scrambled/random controls. If observed ≈ permuted/random, the
benefit is shared-trunk regularization, NOT cytogenetic biology. This SUPPORTS the Fig. 3
falsification result (the primary discrimination matrix) but is not itself the headline —
it is the complementary calibration view.

PLOT-ONLY. Reads outputs/stageD_open_gdc_os/stageD_detail.csv (per-fold) for CIs and
stageD_negative_controls.csv (means). Endpoint = OS technical validation; CNV-derived
subtype labels are sequencing-inferred (NOT FISH).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import COLORS, save, plt

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "outputs" / "stageD_open_gdc_os"
COND_ORDER = ["real", "permuted", "random", "lambda0", "lambda_huge"]
COND_LABEL = {"real": "observed (CNV)", "permuted": "permuted", "random": "random",
              "lambda0": "λ=0 (no distill)", "lambda_huge": "λ→∞"}
COND_COLOR = {"real": COLORS["confirmed"], "permuted": COLORS["blocked"],
              "random": COLORS["caution"], "lambda0": COLORS["model"], "lambda_huge": "#9467bd"}


def _small_subtypes(detail):
    prev = detail.groupby("subtype")["mean_test_n"].mean() if "mean_test_n" in detail else None
    # least-prevalent two CNV subtypes (match experiments_stageD._decide)
    counts = detail.groupby("subtype")["test_n"].mean() if "test_n" in detail else prev
    return list(counts.sort_values().index[:2])


def main():
    detail = pd.read_csv(SD / "stageD_detail.csv")
    detail = detail[detail.get("status", "ok") == "ok"].copy()
    detail["improvement"] = detail["ibs_independent"] - detail["ibs_hss"]
    small = _small_subtypes(detail)
    sub = detail[detail.subtype.isin(small)]

    # per-condition mean ± 95% CI across (subtype,fold) of small subtypes
    stats = {}
    for c in COND_ORDER:
        v = sub[sub.condition == c]["improvement"].dropna().values
        if len(v) == 0:
            continue
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else (v.mean(), 0)
        stats[c] = (m, 1.96 * se, len(v))

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.axhline(0, color="#333333", lw=1.0)
    xs = range(len(stats))
    for x, c in zip(xs, [c for c in COND_ORDER if c in stats]):
        m, ci, n = stats[c]
        ax.errorbar(x, m, yerr=ci, fmt="o", color=COND_COLOR[c], ms=8, capsize=4, lw=2)
        ax.text(x, m + ci + 0.004, f"{m:+.3f}", ha="center", fontsize=8, color=COND_COLOR[c])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER if c in stats], fontsize=8.5)
    ax.set_ylabel("small-subtype IBS improvement\n(independent − HSS; +ve = HSS better)", fontsize=9)
    real_m = stats["real"][0]
    ctrl_max = max(stats["permuted"][0], stats["random"][0])
    ax.set_title(f"Subtype calibration null: observed CNV ({real_m:+.3f}) ≤ scrambled controls "
                 f"(max {ctrl_max:+.3f})\n→ benefit is regularization, not cytogenetic biology (pre-registered STOP)",
                 fontsize=9.5)
    ax.annotate("scrambled labels help\nas much or MORE", xy=(1, stats["permuted"][0]),
                xytext=(1.4, stats["permuted"][0] + 0.02), fontsize=7.5, color=COLORS["blocked"],
                arrowprops=dict(arrowstyle="->", color=COLORS["blocked"], lw=1))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, -0.03, f"Small subtypes: {', '.join(small)} (sequencing-inferred, NOT FISH). "
             "OS technical validation only.", fontsize=6.8, color="#555555")
    save(fig, "sfig5_subtype_calibration_ibs")


if __name__ == "__main__":
    main()
