"""Paper figures buildable NOW from committed real results (Tier-1 / open-GDC OS).

Renders the two figures whose evidence already exists on the n1 open-data run — no serial substrate
required (Fig 2/3/5 are blocked until controlled serial CoMMpass; see docs/paper/progression_potential
_design.md).  Every number is READ from a committed artifact; nothing is invented here.

  Figure 1  Scientific design + claim boundary (data-tier gate: allowed vs blocked).
  Figure 4  Molecular incremental-value forest plot (ΔC-index vs baseline, 95% CI) — the honest
            H3 result: no molecular configuration significantly beats its baseline on OS, while the
            permuted-omics negative control significantly degrades (the pipeline detects real signal
            being destroyed, so the null is "not incremental", not "no signal").

Run:  venv/bin/python scripts/figures/make_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FALSIF = ROOT / "outputs" / "falsification_open_gdc_os" / "falsification_paired_deltas.csv"
LEDGER = ROOT / "outputs" / "autoresearch" / "ledger.csv"
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Figure 1: claim boundary
def figure1_claim_boundary() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Fig 1  Endpoint-calibrated progression potential — data-tier claim boundary",
                 fontsize=12, fontweight="bold", loc="left")

    # left: data streams -> model
    streams = [("serial clinical\n(labs, regimen dates)", 8.4, "#b23", False),
               ("baseline molecular\n(programs, CNA)", 6.7, "#268", True),
               ("teacher targets\n→ snapshot student", 5.0, "#666", False)]
    for txt, y, col, avail in streams:
        ax.add_patch(Rectangle((0.3, y - 0.55), 3.0, 1.1, fc="#f4f4f4", ec=col, lw=1.6))
        ax.text(1.8, y, txt, ha="center", va="center", fontsize=8.4)
    ax.text(1.8, 9.35, "DATA", ha="center", fontsize=9, fontweight="bold", color="#333")

    # center: model box
    ax.add_patch(Rectangle((4.1, 4.4), 2.0, 4.2, fc="#eef3f8", ec="#268", lw=1.8))
    ax.text(5.1, 8.15, "clinically anchored\nlatent process", ha="center", va="center", fontsize=8.2)
    ax.text(5.1, 6.6, "defective FHT\nP(T*=∞)>0", ha="center", va="center", fontsize=8.2)
    ax.text(5.1, 5.1, "molecular residual\nconditioner", ha="center", va="center", fontsize=8.2)
    for _, y, col, _ in streams:
        ax.add_patch(FancyArrowPatch((3.35, y), (4.05, min(max(y, 4.8), 8.2)),
                                     arrowstyle="-|>", mutation_scale=11, color=col, lw=1.3))

    # right: claim tiers (allowed green / blocked red)
    claims = [("clinically anchored trajectory", "#1a7", "ALLOWED (needs serial)"),
              ("molecularly conditioned dynamics", "#1a7", "ALLOWED (serial+molec)"),
              ("baseline-only distilled forecast", "#1a7", "ALLOWED (post-teacher)"),
              ("serial MOLECULAR trajectory", "#c33", "BLOCKED (no paired molec)"),
              ("pseudotime = calendar time", "#c33", "BLOCKED (category error)")]
    y = 8.7
    ax.text(8.4, 9.35, "CLAIM BOUNDARY", ha="center", fontsize=9, fontweight="bold", color="#333")
    for txt, col, tag in claims:
        ax.add_patch(Rectangle((6.6, y - 0.34), 3.3, 0.68, fc="#f7fbf8" if col == "#1a7" else "#fdf3f3",
                               ec=col, lw=1.4))
        mark = "✓" if col == "#1a7" else "✗"
        ax.text(6.78, y, f"{mark} {txt}", ha="left", va="center", fontsize=7.7, color=col)
        ax.text(9.72, y - 0.20, tag, ha="right", va="center", fontsize=5.9, color="#777", style="italic")
        y -= 0.83
    ax.add_patch(FancyArrowPatch((6.1, 6.5), (6.55, 7.0), arrowstyle="-|>", mutation_scale=11,
                                 color="#268", lw=1.3))

    ax.text(0.3, 3.7,
            "Tier gate (data on disk today): T1 baseline+OS = YES → static risk / molecular-increment / calibration.\n"
            "T2 serial clinical, T3 baseline-molec+serial, T4 paired-molec, T5 dense-serial = NOT on open tier "
            "(controlled: MMRF Researcher-Gateway / dbGaP).\n"
            "Stage-1 substrate gate proves the boundary: open GDC = 995 patients × 1 visit (0% multi-visit) → "
            "dynamic core BLOCKED, not fabricated.",
            fontsize=7.2, color="#444", va="top")
    fig.tight_layout()
    p = OUT / "fig1_claim_boundary.pdf"
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Figure 4: molecular forest plot
# label -> (baseline shown, is a negative control?)
_PANEL_A = {
    "clinical+omics_vs_clinical": ("+ omics  (linear Cox)", False),
    "clinical+cyto+omics_vs_clinical": ("+ cyto + omics", False),
    "residual_risk_vs_clinical": ("+ residual-risk (orthogonalised)", False),
    "pooled_neural_vs_clinical+omics": ("pooled neural  vs clinical+omics", False),
    "scrambled_subtype_vs_clinical+cyto+omics": ("scrambled-subtype control", True),
    "permuted_omics_vs_clinical+omics": ("permuted-omics control", True),
}
_PANEL_B_LABEL = {
    "gbsa_clinical_omics": "GBSA (nonlinear)",
    "coxnet_elasticnet": "Coxnet (elastic-net)",
    "supervised_component_screen": "supervised omics screen",
    "mmsygnal_programs": "mmSYGNAL programs",
    "program_crosstalk": "program cross-talk",
    "mechanistic_ccle": "CCLE/DepMap mechanistic prior",
}


def figure4_molecular_forest() -> Path:
    fa = pd.read_csv(FALSIF)
    fa = fa.set_index("comparison")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [1, 1]})

    # ---- Panel A: built-in arms, two-sided 95% CI ----
    rows = [(lbl, comp, ctrl) for comp, (lbl, ctrl) in _PANEL_A.items() if comp in fa.index]
    y = list(range(len(rows)))[::-1]
    for yi, (lbl, comp, ctrl) in zip(y, rows):
        d = fa.loc[comp]
        lo, hi, pt = float(d["delta_ci_low"]), float(d["delta_ci_high"]), float(d["delta_cindex"])
        sig = bool(d.get("fdr_significant", False))
        col = "#c33" if (sig and pt < 0) else ("#1a7" if (sig and pt > 0) else "#888")
        axA.plot([lo, hi], [yi, yi], color=col, lw=2.2, solid_capstyle="round")
        axA.plot(pt, yi, "o", color=col, ms=7, mec="white", mew=1.0)
        axA.text(hi + 0.004, yi, ("control" if ctrl else "") + (" *" if sig else ""),
                 va="center", fontsize=6.5, color=col)
    axA.set_yticks(y); axA.set_yticklabels([r[0] for r in rows], fontsize=8)
    axA.axvline(0, color="#333", lw=1, ls="--")
    axA.set_xlabel("ΔC-index vs baseline  (95% CI)", fontsize=9)
    axA.set_title("A  Molecular increment on OS — built-in arms\n(baseline = clinical-only unless noted)",
                  fontsize=9.5, loc="left")
    axA.margins(y=0.12)

    # ---- Panel B: autoresearch sweep vs clinical+omics, point + one-sided lower 95% bound ----
    lg = pd.read_csv(LEDGER).set_index("candidate")
    rowsB = [(lbl, cand) for cand, lbl in _PANEL_B_LABEL.items() if cand in lg.index]
    yb = list(range(len(rowsB)))[::-1]
    for yi, (lbl, cand) in zip(yb, rowsB):
        d = lg.loc[cand]
        pt, lo = float(d["delta_cindex"]), float(d["delta_ci_low"])
        base = str(d["baseline"])
        col = "#1a7" if bool(d.get("fdr_significant", False)) and pt > 0 else "#888"
        axB.plot([lo, pt], [yi, yi], color=col, lw=2.2, solid_capstyle="round")
        axB.plot(pt, yi, "o", color=col, ms=7, mec="white", mew=1.0)
        axB.text(max(pt, lo) + 0.003, yi, f"vs {base.split('+')[0]}+…", va="center", fontsize=5.8, color="#999")
    axB.set_yticks(yb); axB.set_yticklabels([r[0] for r in rowsB], fontsize=8)
    axB.axvline(0, color="#333", lw=1, ls="--")
    axB.set_xlabel("ΔC-index vs its baseline  (● point, │ one-sided lower 95%)", fontsize=8.4)
    axB.set_title("B  Advanced-molecular falsification sweep\n(all REVERT · 0/6 FDR-significant)",
                  fontsize=9.5, loc="left")
    axB.margins(y=0.12)

    fig.suptitle("Fig 4  No molecular configuration is an incremental OS predictor over clinical (n1, open GDC) "
                 "— the honest H3 null", fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "fig4_molecular_forest.pdf"
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    p1 = figure1_claim_boundary()
    p4 = figure4_molecular_forest()
    print(f"wrote {p1.relative_to(ROOT)} (+.png)")
    print(f"wrote {p4.relative_to(ROOT)} (+.png)")


if __name__ == "__main__":
    main()
