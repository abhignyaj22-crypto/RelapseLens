"""Shared visual style guide for manuscript figures (see docs/figures/figure_specs.md)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "clinical": "#1f6fb2",      # blue
    "cytogenetics": "#7b4fa3",  # purple
    "omics": "#2e8b57",         # green
    "model": "#5a5a5a",         # gray
    "caution": "#e08214",       # orange (claim gate / off-endpoint)
    "blocked": "#d62728",       # red (not claimed)
    "confirmed": "#2ca02c",     # green check
    "hypothesis": "#e6b800",    # yellow triangle
}


def color_for(label: str) -> str:
    l = label.lower()
    if "mmsygnal" in l:
        return COLORS["caution"]      # off-endpoint comparator
    if "omics" in l:
        return COLORS["omics"]
    if "cyto" in l:
        return COLORS["cytogenetics"]
    if "clinical" in l:
        return COLORS["clinical"]
    return COLORS["model"]


def save(fig, stem: str, dpi: int = 300):
    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "figures" / "final"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/final/{stem}.pdf and .png ({dpi} dpi)")
