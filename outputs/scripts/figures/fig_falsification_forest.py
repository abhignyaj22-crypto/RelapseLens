"""Figure — falsification forest plot of paired ΔC-index (PLOT-ONLY).

Reads outputs/falsification_open_gdc_os/falsification_paired_deltas.csv (produced by
`mm-tte falsify`) and draws each arm's paired ΔC vs its pre-specified baseline with the
95% CI, colored by BH-FDR significance. No data are computed here.
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import plt, COLORS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "outputs" / "falsification_open_gdc_os" / "falsification_paired_deltas.csv"


def main():
    if not CSV.exists():
        raise SystemExit(f"missing {CSV} — run `mm-tte falsify` first")
    d = pd.read_csv(CSV)
    d = d.iloc[::-1].reset_index(drop=True)   # top-to-bottom reading order
    y = range(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(d) + 1.4))
    for i, r in d.iterrows():
        sig = bool(r.get("fdr_significant", False))
        col = COLORS["confirmed"] if sig else COLORS["blocked"]
        lo, hi = r["delta_ci_low"], r["delta_ci_high"]
        ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(r["delta_cindex"], i, "o", color=col, ms=7)
    ax.axvline(0.0, color="#888", ls="--", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["comparison"].tolist(), fontsize=9)
    ax.set_xlabel(r"paired $\Delta$C-index (arm $-$ baseline), 95% CI")
    ax.set_title("Falsification matrix: does any structure survive?\n"
                 "green = BH-FDR significant · red = not significant (CI overlaps 0)", fontsize=10)
    ax.margins(y=0.12)
    ax.grid(axis="x", alpha=0.25)
    try:
        from _style import save
        save(fig, "fig_falsification_forest")
    except Exception:
        out = ROOT / "figures" / "final"; out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig_falsification_forest.png", dpi=300, bbox_inches="tight")
        print(f"wrote {out/'fig_falsification_forest.png'}")


if __name__ == "__main__":
    main()
