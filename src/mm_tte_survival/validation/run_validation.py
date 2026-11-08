"""Orchestrate the layered subtype-label validation stack and write a consolidated
claim card. Layers (strictly open data, nothing synthetic):

  1. External real-FISH         (GSE6477: del13 + hyperdiploid)      external_geo.py
  2. External cluster concord.  (GSE19784: translocation surrogates) external_geo.py
  3. Internal cross-modality    (CoMMpass CNV vs orthogonal expr)    internal_concordance.py
  4. Label-noise robustness     (flip at published discordance)      experiments_label_noise.py
  5. FISH-ready harness         (inert until a FISH file is supplied) fish_ready.py
  + Literature concordance      docs/literature_cnv_fish_concordance.md

The residual limitation is stated plainly: CoMMpass calls are NOT validated against
CoMMpass FISH; trust rests on external GEO, literature concordance, internal
corroboration, and demonstrated robustness to realistic label noise.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .external_geo import run_external_validation
from .internal_concordance import run_internal_concordance
from .fish_ready import run_fish_ready


def run_subtype_validation(cfg: dict) -> dict:
    outdir = Path(cfg.get("paths", {}).get("outdir", "outputs/validation"))
    outdir.mkdir(parents=True, exist_ok=True)

    external = run_external_validation(
        cache_dir=cfg.get("paths", {}).get("external_cache", "data/external"),
        outdir=str(outdir))
    internal = run_internal_concordance(
        gene_matrix_npz=cfg.get("paths", {}).get("gene_matrix", "data/real/gene_matrix.npz"),
        cyto_csv=cfg.get("paths", {}).get("cytogenetics", "data/real/cytogenetics.csv"),
        outdir=str(outdir))

    # Label-noise robustness (heavier; needs the full cohort cfg). Optional.
    noise = None
    if cfg.get("run_label_noise", True):
        from ..experiments_label_noise import run_label_noise
        noise = run_label_noise({**cfg, "paths": {**cfg["paths"],
                                                  "outdir": str(outdir / "label_noise")}})

    fish = run_fish_ready(cfg)  # None unless a real FISH file is supplied

    _write_summary(outdir / "subtype_validation_summary.md", external, internal, noise, fish)
    return {"outdir": str(outdir), "external": external, "internal": internal,
            "label_noise": noise, "fish": fish}


def _fish_status(fish) -> str:
    if fish is None or (isinstance(fish, pd.DataFrame) and fish.empty):
        return ("INERT — no CoMMpass FISH file supplied. Drop a controlled-access MMRF "
                "seqFISH file into paths.fish to compute genuine sens/spec/kappa.")
    return "ACTIVE — real CoMMpass FISH supplied; see fish_concordance.csv."


def _write_summary(path, external, internal, noise, fish) -> None:
    fish_rows = external[external["is_fish"] == True]  # noqa: E712
    lines = [
        "# Subtype-label validation — consolidated claim card",
        "",
        "**Question:** are the sequencing-inferred subtype labels trustworthy? No FISH",
        "exists for the open CoMMpass cohort (MMRF seqFISH is controlled-access), so trust",
        "is established with a layered open-data stack. Nothing here is synthetic.",
        "",
        "## 1. External real-FISH (GSE6477)",
        "del(13) and hyperdiploidy validated against genuine interphase FISH "
        "(expression-only array; cross-platform caveat).",
        _mini(fish_rows, ["subtype", "auc", "sens", "spec", "kappa"]),
        "",
        "## 2. External cluster concordance (GSE19784, NOT FISH)",
        "translocation surrogates vs the published expression cluster.",
        _mini(external[external["is_fish"] == False], ["subtype", "auc", "kappa"]),  # noqa: E712
        "",
        "## 3. Internal cross-modality concordance (CoMMpass, NOT FISH)",
        "CNV calls vs orthogonal RNA dosage.",
        _mini(internal, ["subtype", "auc", "pointbiserial_r", "kappa"]),
        "",
        "## 4. Label-noise robustness",
        (noise["decision"]["verdict"] if noise else "_not run_"),
        "",
        "## 5. FISH-ready harness",
        _fish_status(fish),
        "",
        "## Residual limitation (stated plainly)",
        "CoMMpass calls are NOT validated against CoMMpass FISH. del(17p)/amp(1q)/del(1p)",
        "have no open FISH and rest on the literature concordance table",
        "(docs/literature_cnv_fish_concordance.md). Strongest support: del(13) (external",
        "FISH + internal); weakest: del(1p), and t(14;16) (surrogate fails the cluster",
        "check). Claims are scoped accordingly; translocations remain exploratory.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def _mini(df: pd.DataFrame, cols) -> str:
    if df is None or df.empty:
        return "_no rows_"
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    for c in cols[1:]:
        d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])
