#!/usr/bin/env python
"""Step 2a: download real GDC open-access Copy Number Segment files for the cohort.

These are GRCh38 WGS copy-ratio segment files (.cr.igv.seg-style: chrom, start,
end, num_probes, seg.mean = log2 copy ratio). Open access. We map each file to a
patient (cases.submitter_id), pick one sample per patient, and bulk-download.
"""
from __future__ import annotations
import json, urllib.parse, urllib.request, sys, io, tarfile, gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEGDIR = ROOT / "data" / "real" / "cnv_seg"
MAPCSV = ROOT / "data" / "real" / "cnv_file_map.tsv"
API = "https://api.gdc.cancer.gov"

FILT = {"op": "and", "content": [
    {"op": "in", "content": {"field": "cases.project.project_id", "value": ["MMRF-COMMPASS"]}},
    {"op": "in", "content": {"field": "data_type", "value": ["Copy Number Segment"]}},
    {"op": "in", "content": {"field": "access", "value": ["open"]}},
    {"op": "in", "content": {"field": "data_format", "value": ["txt"]}},
]}
FIELDS = "file_id,file_name,cases.submitter_id,cases.samples.sample_type,cases.samples.submitter_id"


def list_files() -> list[dict]:
    out, frm, size = [], 0, 200
    while True:
        params = {"filters": json.dumps(FILT), "fields": FIELDS,
                  "format": "JSON", "size": str(size), "from": str(frm)}
        url = API + "/files?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=180) as r:
            d = json.load(r)
        out.extend(d["data"]["hits"])
        total = d["data"]["pagination"]["total"]
        frm += size
        print(f"  listed {len(out)}/{total}", file=sys.stderr)
        if frm >= total:
            break
    return out


def pick_one_per_patient(hits: list[dict]) -> dict[str, dict]:
    """One CNV file per patient; prefer a primary/tumor bone-marrow sample."""
    chosen: dict[str, dict] = {}
    for h in hits:
        case = (h.get("cases") or [{}])[0]
        pid = case.get("submitter_id")
        if not pid:
            continue
        samples = case.get("samples") or [{}]
        stype = (samples[0].get("sample_type") or "")
        rec = {"file_id": h["file_id"], "file_name": h["file_name"],
               "sample_type": stype, "sample_id": samples[0].get("submitter_id", "")}
        # prefer the lexicographically-first sample id for determinism (baseline = _1_)
        if pid not in chosen or rec["sample_id"] < chosen[pid]["sample_id"]:
            chosen[pid] = rec
    return chosen


def bulk_download(file_ids: list[str], dest: Path):
    """POST to /data to retrieve a tar of many files; extract .seg.txt to dest."""
    dest.mkdir(parents=True, exist_ok=True)
    batch = 200
    for i in range(0, len(file_ids), batch):
        chunk = file_ids[i:i + batch]
        payload = json.dumps({"ids": chunk}).encode()
        req = urllib.request.Request(API + "/data", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            blob = r.read()
        # response is a tar (single file) or tar.gz (multi). Try gzip then tar.
        try:
            raw = gzip.decompress(blob)
        except OSError:
            raw = blob
        try:
            tf = tarfile.open(fileobj=io.BytesIO(raw))
            members = tf.getmembers()
            for m in members:
                if m.isfile() and m.name.endswith(".txt"):
                    fid = m.name.split("/")[0]
                    data = tf.extractfile(m).read()
                    (dest / f"{fid}.seg.txt").write_bytes(data)
        except tarfile.ReadError:
            # single-file (one id) non-tar response
            (dest / f"{chunk[0]}.seg.txt").write_bytes(raw)
        print(f"  downloaded {min(i+batch, len(file_ids))}/{len(file_ids)}", file=sys.stderr)


def main():
    hits = list_files()
    chosen = pick_one_per_patient(hits)
    print(f"patients with CNV: {len(chosen)}")
    MAPCSV.parent.mkdir(parents=True, exist_ok=True)
    with MAPCSV.open("w") as f:
        f.write("patient_id\tfile_id\tfile_name\tsample_type\tsample_id\n")
        for pid, rec in sorted(chosen.items()):
            f.write(f"{pid}\t{rec['file_id']}\t{rec['file_name']}\t{rec['sample_type']}\t{rec['sample_id']}\n")
    bulk_download([r["file_id"] for r in chosen.values()], SEGDIR)
    n = len(list(SEGDIR.glob("*.seg.txt")))
    print(f"wrote {n} seg files to {SEGDIR}; map -> {MAPCSV}")
    # fail loud on a partial cohort — swallowed per-file errors must not pass as complete
    assert n == len(chosen), (
        f"partial CNV cohort: {n} seg files on disk but {len(chosen)} patients expected. "
        f"Some downloads errored (OSError/ReadError above); re-run to complete.")


if __name__ == "__main__":
    main()
