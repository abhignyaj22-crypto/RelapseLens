#!/usr/bin/env python
"""Figure 3 (intellectual center) — the falsification matrix.

Subtype-aware / molecular conditioning does NOT survive pre-specified negative controls in
open-GDC OS. Three panels driven by the Phase-3 falsification matrix:

  Panel A  absolute discrimination (Uno IPCW C) per arm, mean +/- 95% CI over splits
  Panel B  paired deltas vs the relevant control/reference (forest centered at 0)
  Panel C  gate-verdict table (the discovery, stated explicitly)

Claim-safe language: CNV-derived subtype labels are sequencing-inferred (NOT FISH); we say
"CNV-derived / observed subtype", never "real subtype". OS technical validation only.

PLOT-ONLY. Canonical input: outputs/falsification_open_gdc_os/falsification_per_split.csv
(long format: split_id, arm, plot_label, metric, value, endpoint, label_source, control_type).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import COLORS, save, plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "outputs" / "falsification_open_gdc_os" / "falsification_per_split.csv"

# Panel A display order (arm key -> short label). Never "real subtype".
PANEL_A = [
    ("clinical",            "Penalized Cox (ref.)"),
    ("clinical+omics",      "Clinical + omics Cox"),
    ("pooled_neural",       "Pooled neural"),
    ("clinical+cyto+omics", "CNV-derived subtype"),
    ("scrambled_subtype",   "Scrambled-subtype (ctrl)"),
    ("permuted_omics",      "Permuted-omics (ctrl)"),
    ("random_label",        "Random-label (ctrl)"),
]
CTRL_ARMS = {"scrambled_subtype", "permuted_omics", "random_label"}


def _c(df, arm):
    return df[(df.arm == arm) & (df.metric == "uno_cindex")].set_index("split_id")["value"]


def _mean_ci(v):
    v = np.asarray(pd.Series(v).dropna())
    if len(v) == 0:
        return np.nan, np.nan, 0
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), 1.96 * se, len(v)


def _paired(df, a, b):
    va, vb = _c(df, a), _c(df, b)
    idx = va.index.intersection(vb.index)
    return _mean_ci((va.loc[idx] - vb.loc[idx]).values)


def main():
    if not CSV.exists():
        raise SystemExit(f"missing {CSV} — run `make benchmark` (mm-tte falsify) first")
    df = pd.read_csv(CSV)

    fig = plt.figure(figsize=(8.2, 9.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.82, 0.86], hspace=0.9)
    axA, axB, axC = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # ---- Panel A: absolute C per arm ----
    ys = list(range(len(PANEL_A)))[::-1]
    for y, (arm, lab) in zip(ys, PANEL_A):
        m, ci, n = _mean_ci(_c(df, arm))
        col = COLORS["blocked"] if arm in CTRL_ARMS else COLORS["clinical"]
        axA.errorbar(m, y, xerr=ci, fmt="o", color=col, ms=6, capsize=3, lw=1.8)
    axA.axvline(0.5, color="#999", ls=":", lw=1, label="chance")
    axA.set_yticks(ys); axA.set_yticklabels([l for _, l in PANEL_A], fontsize=8)
    axA.set_xlabel("Uno IPCW C-index (mean $\\pm$95% CI over splits)", fontsize=8.5)
    axA.set_title("A  Discrimination across arms", fontsize=10, loc="left")
    axA.legend(fontsize=7, loc="lower right"); axA.margins(y=0.08)

    # ---- Panel B: paired deltas vs control/reference ----
    DELTAS = [
        ("Clinical+omics $-$ Penalized Cox\n(incremental value)", "clinical+omics", "clinical"),
        ("Pooled neural $-$ Clinical+omics Cox\n(nonlinear gain)", "pooled_neural", "clinical+omics"),
        ("CNV-subtype $-$ Scrambled subtype\n(subtype biology)",   "clinical+cyto+omics", "scrambled_subtype"),
        ("Clinical+omics $-$ Permuted omics\n(molecular signal)",  "clinical+omics", "permuted_omics"),
    ]
    yb = list(range(len(DELTAS)))[::-1]
    for y, (lab, a, b) in zip(yb, DELTAS):
        m, ci, n = _paired(df, a, b)
        robust = np.isfinite(m) and (m - ci) > 0
        col = COLORS["confirmed"] if robust else COLORS["blocked"]
        axB.errorbar(m, y, xerr=ci, fmt="o", color=col, ms=6, capsize=3, lw=1.8)
    axB.axvline(0.0, color="#333", ls="--", lw=1)
    axB.set_yticks(yb); axB.set_yticklabels([l for l, _, _ in DELTAS], fontsize=7.2)
    axB.set_xlabel("paired $\\Delta$ Uno-C (arm $-$ control), 95% CI", fontsize=8.5)
    axB.set_title("B  Does any arm beat its control?", fontsize=10, loc="left")
    axB.margins(y=0.14)

    # ---- Panel C: gate-verdict table ----
    axC.axis("off")
    def verdict(a, b):
        m, ci, n = _paired(df, a, b)
        lo = m - ci
        return ("PASS" if lo > 0 else "FAIL"), f"$\\Delta$={m:+.3f} [{lo:+.3f}]"
    rc_m, _, _ = _mean_ci(_c(df, "random_label"))
    rows = [
        ("Clinical+omics > Penalized Cox? (incremental value)", *verdict("clinical+omics", "clinical")),
        ("Pooled neural > Clinical+omics Cox? (nonlinear gain)", *verdict("pooled_neural", "clinical+omics")),
        ("CNV-subtype > Scrambled subtype? (biology)",          *verdict("clinical+cyto+omics", "scrambled_subtype")),
        ("Clinical+omics > Permuted omics? (signal specificity)", *verdict("clinical+omics", "permuted_omics")),
        ("Random-label near chance? (leakage sanity)",
         ("PASS" if abs(rc_m - 0.5) < 0.06 else "WARN"), f"C={rc_m:.3f} $\\approx$ 0.5"),
    ]
    cell, cols = [], ["Falsification test", "Verdict", "Effect"]
    colours = []
    for test, v, detail in rows:
        cell.append([test, v, detail])
        colours.append(COLORS["confirmed"] if v == "PASS" and "sanity" in test else
                       (COLORS["confirmed"] if v == "PASS" else COLORS["blocked"]))
    tbl = axC.table(cellText=cell, colLabels=cols, loc="center", cellLoc="left",
                    colWidths=[0.60, 0.13, 0.27])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.35)
    for i, v_color in enumerate(colours, start=1):
        tbl[(i, 1)].get_text().set_color(v_color); tbl[(i, 1)].get_text().set_weight("bold")
    for c in range(3):
        tbl[(0, c)].get_text().set_weight("bold")
    axC.set_title("C  Gate verdicts — no arm shows robust incremental biological value",
                  fontsize=10, loc="left")

    fig.suptitle("Fig. 3  Subtype-aware / molecular conditioning does not survive "
                 "falsification controls (open-GDC OS)", fontsize=11, y=1.0)
    fig.text(0.5, -0.02,
             "CNV-derived subtype labels are sequencing-inferred (NOT FISH). OS technical "
             "validation only — no PFS / relapse / treatment-resistance / clinical-deployment claim.",
             ha="center", fontsize=6.8, color="#555")
    save(fig, "fig3_falsification_matrix")


if __name__ == "__main__":
    main()
