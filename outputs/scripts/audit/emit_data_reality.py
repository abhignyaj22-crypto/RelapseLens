#!/usr/bin/env python3
"""Stage A: sample every data file (header + ≥5 rows) and emit docs/DATA_REALITY.md.

Read-only audit — never trusts data_manifest.json without reconciling against disk.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "docs" / "DATA_REALITY.md"
MANIFEST = DATA / "real" / "data_manifest.json"

MODALITY_HINTS = {
    "clinical_survival": ("clinical", "patient"),
    "clinical": ("clinical", "patient"),
    "clinical_serial": ("clinical", "patient"),
    "cytogenetics": ("cytogenetics", "patient"),
    "omics": ("transcriptomics", "patient"),
    "program_activity": ("program_activity", "patient"),
    "mmsygnal": ("program_activity", "patient"),
    "gene_matrix": ("transcriptomics", "patient"),
    "rna_counts": ("transcriptomics", "patient"),
    "cnv_seg": ("copy_number", "patient"),
    "GSE": ("transcriptomics", "patient"),
    "GPL": ("reference", "network"),
    "series_matrix": ("transcriptomics", "patient"),
}


def _infer_modality(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    for key, (mod, dom) in MODALITY_HINTS.items():
        if key in name or key in str(path.parent).lower():
            return mod, dom
    if path.suffix in {".seg.txt"}:
        return "copy_number", "patient"
    if path.suffix in {".star.tsv"}:
        return "transcriptomics", "patient"
    if path.suffix in {".h5", ".h5ad"}:
        return "scrna", "patient"
    return "unknown", "unknown"


def _sample_csv(path: Path, n: int = 5) -> dict:
    df = pd.read_csv(path, nrows=n + 100)
    head = df.head(n)
    return {
        "shape": list(df.shape) if len(df) < 200 else _full_shape(path),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns[:30]},
        "sample_rows": head.to_dict(orient="records"),
    }


def _full_shape(path: Path) -> list[int]:
    if path.suffix == ".tsv":
        ncols = len(pd.read_csv(path, nrows=0, sep="\t").columns)
        with open(path) as f:
            nrows = sum(1 for _ in f) - 1
        return [nrows, ncols]
    df = pd.read_csv(path)
    return list(df.shape)


def _sample_tsv(path: Path, n: int = 5) -> dict:
    df = pd.read_csv(path, sep="\t", nrows=n + 100)
    head = df.head(n)
    shape = _full_shape(path)
    return {
        "shape": shape,
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns[:30]},
        "sample_rows": head.to_dict(orient="records"),
    }


def _sample_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    info = {"keys": list(z.files)}
    for k in z.files:
        arr = z[k]
        info[f"{k}_shape"] = list(arr.shape)
        info[f"{k}_dtype"] = str(arr.dtype)
        if k in ("patient_ids", "genes") and arr.ndim == 1:
            info[f"{k}_sample"] = [str(x) for x in arr[:5]]
        elif arr.ndim == 2:
            info[f"{k}_sample"] = arr[:2, :5].tolist()
    return info


def _sample_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sample_file(path: Path) -> dict:
    rel = path.relative_to(DATA)
    mod, dom = _infer_modality(path)
    entry: dict = {"path": str(rel), "modality": mod, "domain": dom, "bytes": path.stat().st_size}
    suf = path.suffix.lower()
    try:
        if suf == ".json":
            entry["content"] = _sample_json(path)
        elif suf == ".npz":
            entry.update(_sample_npz(path))
        elif suf in {".tsv", ".txt"} and "seg" in path.name:
            entry.update(_sample_tsv(path))
        elif suf in {".csv", ".tsv"}:
            sep = "\t" if suf == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, nrows=5)
            full = _full_shape(path) if path.stat().st_size < 50_000_000 else None
            entry.update({
                "shape": full or [5, len(df.columns)],
                "columns": list(df.columns),
                "dtypes": {c: str(df[c].dtype) for c in df.columns[:30]},
                "sample_rows": df.head(5).to_dict(orient="records"),
            })
        elif suf == ".gz":
            entry["note"] = "compressed — header sampled via pandas"
            df = pd.read_csv(path, compression="gzip", nrows=5)
            entry.update({
                "columns": list(df.columns)[:20],
                "sample_rows": df.head(5).to_dict(orient="records"),
            })
        elif path.stat().st_size == 0:
            entry["note"] = "EMPTY FILE"
        else:
            entry["note"] = f"unhandled suffix {suf}"
    except Exception as e:
        entry["error"] = str(e)
    return entry


def _collect_files() -> list[Path]:
    files = []
    for sub in ("real", "external", "demo", "schema"):
        d = DATA / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                files.append(p)
    return files


def _reconcile_manifest(entries: list[dict]) -> list[str]:
    drift = []
    if not MANIFEST.exists():
        drift.append("data_manifest.json missing")
        return drift
    manifest = json.loads(MANIFEST.read_text())
    claimed = set(manifest.get("tables", {}).keys())
    on_disk = {e["path"].split("/")[-1] for e in entries if e["path"].startswith("real/") and "/" not in e["path"][5:]}
    for c in claimed:
        if c not in on_disk:
            drift.append(f"manifest claims {c} but not on disk")
    for e in entries:
        name = e["path"].split("/")[-1]
        if e["path"].startswith("real/") and "/" not in e["path"][5:] and name not in claimed:
            drift.append(f"on disk but not in manifest: {name}")
    return drift


def _format_entry(e: dict) -> str:
    lines = [f"### `{e['path']}`", f"- **modality:** {e['modality']} | **domain:** {e['domain']} | **bytes:** {e['bytes']}"]
    if "shape" in e:
        lines.append(f"- **shape:** {e['shape']}")
    if "columns" in e:
        cols = e["columns"]
        shown = cols[:15]
        lines.append(f"- **columns ({len(cols)}):** `{', '.join(str(c) for c in shown)}{'...' if len(cols) > 15 else ''}`")
    if "keys" in e:
        lines.append(f"- **npz keys:** {e['keys']}")
        for k in e.get("keys", []):
            if f"{k}_shape" in e:
                lines.append(f"  - `{k}`: shape={e[f'{k}_shape']}, dtype={e.get(f'{k}_dtype','?')}")
    if "sample_rows" in e:
        lines.append("- **sample (5 rows):**")
        lines.append("```json")
        lines.append(json.dumps(e["sample_rows"][:5], indent=2, default=str)[:3000])
        lines.append("```")
    if "note" in e:
        lines.append(f"- **note:** {e['note']}")
    if "error" in e:
        lines.append(f"- **ERROR:** {e['error']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    files = _collect_files()
    # For large dirs (rna_counts, cnv_seg), sample one representative file each
    seen_dirs: set[str] = set()
    entries: list[dict] = []
    dir_counts: dict[str, int] = {}
    for p in files:
        parent = str(p.parent.relative_to(DATA))
        if parent in ("real/rna_counts", "real/cnv_seg"):
            dir_counts[parent] = dir_counts.get(parent, 0) + 1
            if parent in seen_dirs:
                continue
            seen_dirs.add(parent)
        entries.append(_sample_file(p))

    drift = _reconcile_manifest(entries)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# DATA REALITY — Stage A audit",
        "",
        f"**Generated:** {ts} by `scripts/audit/emit_data_reality.py`",
        "**Root:** `mm_tte_opisd/data/`",
        "",
        "## Headline",
        "",
        "Patient-domain CoMMpass OPEN data is on disk (transcriptomics, clinical, cytogenetics, programs).",
        "Four ResistanceMap modalities (proteomics, epigenomics, scRNA, PPI) are **absent** — Stage B fetch required.",
        "Proteomics/epigenomics enter as **cell-line surrogates** (CCLE), never patient measurements.",
        "",
        "## Provenance corrections",
        "",
        "- **GSE124310 = Zavidij/Ghobrial, Nat Cancer 2020** (NOT Ledergor). Ledergor 2018 ≈ GSE117156.",
        "",
        "## Manifest drift",
        "",
    ]
    if drift:
        lines.extend(f"- {d}" for d in drift)
    else:
        lines.append("- No drift detected (or manifest missing).")
    lines.append("")
    lines.append(f"## File inventory ({len(files)} files on disk; {len(entries)} sampled)")
    lines.append("")

    by_sub: dict[str, list] = {}
    for e in entries:
        sub = e["path"].split("/")[0]
        by_sub.setdefault(sub, []).append(e)

    for sub, sub_entries in sorted(by_sub.items()):
        lines.append(f"## `data/{sub}/`")
        lines.append("")
        if sub == "real":
            for d, c in sorted(dir_counts.items()):
                lines.append(f"- `{d}/`: {c} files (one representative sampled)")
            lines.append("")
        for e in sub_entries:
            lines.append(_format_entry(e))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT} ({len(entries)} entries from {len(files)} files)")
    if drift:
        print("Drift flags:", len(drift))
    return 0


if __name__ == "__main__":
    sys.exit(main())
