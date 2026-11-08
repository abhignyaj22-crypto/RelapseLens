#!/usr/bin/env python
"""Step 3a: download real GDC open-access STAR gene-counts for the cohort.

One Gene Expression Quantification file per patient (baseline sample preferred).
Resumable: files already on disk are skipped. Downloads are streamed per-file
to keep memory flat.
"""
from __future__ import annotations
import json, urllib.parse, urllib.request, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RNADIR = ROOT / "data" / "real" / "rna_counts"
MAPCSV = ROOT / "data" / "real" / "rna_file_map.tsv"
API = "https://api.gdc.cancer.gov"

FILT = {"op": "and", "content": [
    {"op": "in", "content": {"field": "cases.project.project_id", "value": ["MMRF-COMMPASS"]}},
    {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
    {"op": "in", "content": {"field": "access", "value": ["open"]}},
]}
FIELDS = "file_id,file_name,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type"


def list_files():
    out, frm, size = [], 0, 200
    while True:
        params = {"filters": json.dumps(FILT), "fields": FIELDS,
                  "format": "JSON", "size": str(size), "from": str(frm)}
        with urllib.request.urlopen(API + "/files?" + urllib.parse.urlencode(params), timeout=180) as r:
            d = json.load(r)
        out.extend(d["data"]["hits"])
        total = d["data"]["pagination"]["total"]
        frm += size
        print(f"  listed {len(out)}/{total}", file=sys.stderr)
        if frm >= total:
            break
    return out


def pick_one_per_patient(hits):
    chosen = {}
    for h in hits:
        case = (h.get("cases") or [{}])[0]
        pid = case.get("submitter_id")
        if not pid:
            continue
        samp = (case.get("samples") or [{}])[0]
        sid = samp.get("submitter_id", "")
        rec = {"file_id": h["file_id"], "file_name": h["file_name"], "sample_id": sid}
        if pid not in chosen or sid < chosen[pid]["sample_id"]:
            chosen[pid] = rec
    return chosen


def download_one(fid: str, dest: Path):
    url = f"{API}/data/{fid}"
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    dest.write_bytes(data)


def main():
    RNADIR.mkdir(parents=True, exist_ok=True)
    hits = list_files()
    chosen = pick_one_per_patient(hits)
    print(f"patients with RNA: {len(chosen)}")
    with MAPCSV.open("w") as f:
        f.write("patient_id\tfile_id\tfile_name\tsample_id\n")
        for pid, rec in sorted(chosen.items()):
            f.write(f"{pid}\t{rec['file_id']}\t{rec['file_name']}\t{rec['sample_id']}\n")
    done = 0
    for i, (pid, rec) in enumerate(sorted(chosen.items()), 1):
        dest = RNADIR / f"{rec['file_id']}.star.tsv"
        if dest.exists() and dest.stat().st_size > 1000:
            done += 1
            continue
        try:
            download_one(rec["file_id"], dest)
            done += 1
        except Exception as e:
            print(f"  WARN {pid} {rec['file_id']}: {e}", file=sys.stderr)
        if i % 50 == 0:
            print(f"  downloaded {done}/{len(chosen)}", file=sys.stderr)
    n = len(list(RNADIR.glob("*.star.tsv")))
    print(f"wrote {n} RNA files to {RNADIR}; map -> {MAPCSV}")
    # fail loud on a partial cohort — a silently-truncated download must not pass as complete
    assert n == len(chosen), (
        f"partial RNA cohort: {n} files on disk but {len(chosen)} patients expected. "
        f"Downloads were swallowed per-file (see WARN lines above); re-run to complete.")


if __name__ == "__main__":
    main()
