#!/usr/bin/env python
"""Step 2b: call cytogenetic CNAs from real GDC Copy Number Segment files.

CNV-derivable calls (length-weighted mean log2 copy ratio over GRCh38 cytoband
regions, thresholded):
    amp1q, del1p, del13q, del17p, hyperdiploid

IGH translocations t(4;14), t(11;14), t(14;16) are NOT copy-number events and
cannot be called from segment files; they are emitted empty here and filled with
expression surrogates in Step 3 (build_omics.py) where RNA is available.

Population frequencies are printed and should be sanity-checked against MM
literature (del13 ~45-50%, amp1q ~35-40%, del1p ~20-30%, del17p ~8-12%,
hyperdiploid ~50-55%).
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEGDIR = ROOT / "data" / "real" / "cnv_seg"
MAPCSV = ROOT / "data" / "real" / "cnv_file_map.tsv"
OUT = ROOT / "data" / "real" / "cytogenetics.csv"

GAIN = 0.10    # log2 copy-ratio threshold for gain (vs per-sample baseline)
LOSS = -0.25   # log2 copy-ratio threshold for loss (vs per-sample baseline)
MIN_COV = 0.30  # min fraction of region covered by segments to make a call

# GRCh38 regions (bp). Arm boundaries approximate to the centromere.
REGIONS = {
    "del1p":  ("chr1", 1, 121_500_000, "loss"),
    "amp1q":  ("chr1", 143_200_000, 248_956_422, "gain"),
    "del13q": ("chr13", 17_700_000, 114_364_328, "loss"),
    "del17p": ("chr17", 1, 22_200_000, "loss"),
}
# Hyperdiploidy: whole-chromosome gains of the classic odd chromosomes.
HRD_CHROMS = ["chr3", "chr5", "chr7", "chr9", "chr11", "chr15", "chr19", "chr21"]
CHROM_LEN = {  # GRCh38 lengths
    "chr3": 198_295_559, "chr5": 181_538_259, "chr7": 159_345_973,
    "chr9": 138_394_717, "chr11": 135_086_622, "chr15": 101_991_189,
    "chr19": 58_617_616, "chr21": 46_709_983,
}


def load_segments(path: Path):
    segs = []
    with path.open() as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)
        for row in r:
            if len(row) < 6:
                continue
            _, chrom, start, end, _np, mean = row[:6]
            try:
                segs.append((chrom, int(start), int(end), float(mean)))
            except ValueError:
                continue
    return segs


def region_weighted_mean(segs, chrom, lo, hi):
    """Length-weighted mean log2 ratio over [lo,hi]; returns (mean, covered_frac)."""
    tot_w, acc = 0.0, 0.0
    for c, s, e, m in segs:
        if c != chrom:
            continue
        a, b = max(s, lo), min(e, hi)
        if b <= a:
            continue
        w = b - a
        tot_w += w
        acc += w * m
    span = hi - lo
    if tot_w == 0:
        return None, 0.0
    return acc / tot_w, tot_w / span


def genome_baseline(segs) -> float:
    """Per-sample baseline = length-weighted autosomal median log2 ratio.

    Corrects the small global offset / ploidy shift so loss/gain thresholds are
    taken relative to each sample's diploid baseline rather than absolute 0.
    """
    import numpy as np
    vals, ws = [], []
    for c, s, e, m in segs:
        if c in ("chrX", "chrY", "chrM"):
            continue
        vals.append(m)
        ws.append(e - s)
    if not vals:
        return 0.0
    vals = np.asarray(vals)
    ws = np.asarray(ws, dtype=float)
    order = np.argsort(vals)
    vals, ws = vals[order], ws[order]
    cw = np.cumsum(ws)
    return float(vals[np.searchsorted(cw, cw[-1] / 2.0)])


def call_file(segs) -> dict:
    out = {}
    base = genome_baseline(segs)
    for name, (chrom, lo, hi, direction) in REGIONS.items():
        mean, cov = region_weighted_mean(segs, chrom, lo, hi)
        if mean is not None:
            mean -= base  # relative to per-sample diploid baseline
        if mean is None or cov < MIN_COV:
            out[name] = ""
            continue
        if direction == "loss":
            out[name] = 1 if mean < LOSS else 0
        else:
            out[name] = 1 if mean > GAIN else 0
    # hyperdiploid: count whole-chromosome gains
    gains = 0
    measured = 0
    for chrom in HRD_CHROMS:
        mean, cov = region_weighted_mean(segs, chrom, 1, CHROM_LEN[chrom])
        if mean is None or cov < MIN_COV:
            continue
        measured += 1
        if mean - base > GAIN:
            gains += 1
    out["hyperdiploid"] = 1 if gains >= 2 else (0 if measured >= 4 else "")
    out["_hrd_trisomies"] = gains
    return out


def main():
    pid2file = {}
    with MAPCSV.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            pid2file[row["patient_id"]] = row["file_id"]

    cols = ["patient_id", "amp1q", "del1p", "del13q", "del17p",
            "t_4_14", "t_11_14", "t_14_16", "hyperdiploid"]
    rows = []
    freq = {c: 0 for c in ["amp1q", "del1p", "del13q", "del17p", "hyperdiploid"]}
    denom = {c: 0 for c in freq}
    missing = 0
    for pid, fid in sorted(pid2file.items()):
        path = SEGDIR / f"{fid}.seg.txt"
        if not path.exists():
            missing += 1
            continue
        calls = call_file(load_segments(path))
        row = {
            "patient_id": pid,
            "amp1q": calls.get("amp1q", ""),
            "del1p": calls.get("del1p", ""),
            "del13q": calls.get("del13q", ""),
            "del17p": calls.get("del17p", ""),
            "t_4_14": "",   # expression surrogate added in Step 3
            "t_11_14": "",
            "t_14_16": "",
            "hyperdiploid": calls.get("hyperdiploid", ""),
        }
        rows.append(row)
        for c in freq:
            if isinstance(row[c], int):
                denom[c] += 1
                freq[c] += row[c]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}  N={len(rows)} patients (missing seg files={missing})")
    print("CNV-derived population frequencies (sanity vs MM literature):")
    for c in ["del13q", "amp1q", "hyperdiploid", "del1p", "del17p"]:
        if denom[c]:
            print(f"  {c:13s}: {freq[c]/denom[c]*100:5.1f}%  (n={denom[c]})")


if __name__ == "__main__":
    main()
