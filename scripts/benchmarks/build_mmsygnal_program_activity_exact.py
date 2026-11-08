#!/usr/bin/env python
"""Build mmSYGNAL 141-program activity using the OFFICIAL miner3 pipeline.

This is NOT an approximation from the JSONs: it uses the upstream miner3 functions
exactly as the canonical `bin/miner3-survival` script does:

  exp_data  = miner.correct_batch_effects(miner.remove_null_rows(raw_tpm), True)
  programs  = { p : union of genes across the program's member regulons }   # miner3-survival lines 131-138
  activity  = miner.generateRegulonActivity(programs, exp_data, p=0.05)      # -> {-1,0,1} per program/sample
  out       = activity.T  with columns 0..140

Validation gate: refuses to emit a degenerate matrix (e.g., all +1 or all 0,
which indicates a preprocessing/version mismatch). A non-degenerate matrix means
the official method ran; it is still NOT bit-validated against the upstream IA12
reference (that requires the original training expression), so the downstream
mmSYGNAL comparison must be reported as method-reproduced research benchmark only.

Usage: python build_mmsygnal_program_activity_exact.py [out.csv]
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MINER = ROOT / "external" / "miner3"
MM = ROOT / "external" / "mmSYGNAL-risk-prediction-models" / "data"
RNADIR = ROOT / "data" / "real" / "rna_counts"
MAP = ROOT / "data" / "real" / "rna_file_map.tsv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "real" / "mmsygnal_program_activity_0_140.csv"

sys.path.insert(0, str(MINER))
import numpy as np
import pandas as pd
from miner import miner


def parse_star_tpm(path: Path) -> dict:
    out = {}
    with path.open() as f:
        hdr = None
        for line in f:
            if line.startswith("#"):
                continue
            a = line.rstrip("\n").split("\t")
            if hdr is None and a[0] == "gene_id":
                hdr = {c: i for i, c in enumerate(a)}
                continue
            if hdr is None or a[0].startswith("N_"):
                continue
            try:
                out[a[hdr["gene_id"]].split(".")[0]] = float(a[hdr["tpm_unstranded"]])
            except (KeyError, ValueError):
                pass
    return out


def main():
    programs = json.loads((MM / "transcriptional_programs.json").read_text())
    regulons = json.loads((MM / "regulons.json").read_text())
    # program gene-set = union of member-regulon genes (miner3-survival lines 131-138)
    genesets = {p: sorted({g for r in regs for g in regulons[r]}) for p, regs in programs.items()}
    print(f"programs: {len(genesets)} (labels {min(genesets,key=int)}..{max(genesets,key=int)})", flush=True)

    fmap = pd.read_csv(MAP, sep="\t")
    cols = {}
    for _, r in fmap.iterrows():
        p = RNADIR / f"{r['file_id']}.star.tsv"
        if p.exists():
            cols[r["patient_id"]] = parse_star_tpm(p)
    raw = pd.DataFrame(cols)
    print(f"raw expression: {raw.shape[0]} genes x {raw.shape[1]} samples", flush=True)

    # OFFICIAL miner3 preprocessing + activity
    raw = miner.remove_null_rows(raw)
    exp_data = miner.correct_batch_effects(raw, do_preprocess_tpm=True)
    avail = set(exp_data.index)
    gs = {p: [g for g in genes if g in avail] for p, genes in genesets.items()}
    cov = float(np.mean([len(gs[p]) / max(len(genesets[p]), 1) for p in genesets]))
    print(f"mean program gene coverage: {cov:.3f}", flush=True)

    act = miner.generateRegulonActivity(gs, exp_data, p=0.05)   # programs x samples in {-1,0,1}
    out = act.T                                                  # samples x programs
    out.columns = [str(i) for i in act.index]                   # program labels 0..140
    out.insert(0, "patient_id", out.index)

    v = act.values.astype(float)
    vals = set(np.unique(v))
    mean = float(v.mean())
    frac = {k: float(np.mean(v == k)) for k in (-1.0, 0.0, 1.0)}
    print(f"distribution: unique={sorted(vals)} mean={mean:.3f} frac={frac}", flush=True)

    # ---- non-degenerate validation gate (fail-closed) ----
    degenerate = (not {-1.0, 1.0}.issubset(vals)) or max(frac.values()) > 0.95
    if degenerate:
        sys.exit("BLOCKED: degenerate program activity (preprocessing/version mismatch). "
                 "Refusing to emit. The mmSYGNAL benchmark stays blocked.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({out.shape[0]} patients x 141 programs)", flush=True)
    print("CAVEAT: official miner3 method reproduced, NOT bit-validated vs upstream "
          "IA12 reference. Report mmSYGNAL comparison as method-reproduced research "
          "benchmark only (OS endpoint; no relapse/PFS claim).", flush=True)


if __name__ == "__main__":
    main()
