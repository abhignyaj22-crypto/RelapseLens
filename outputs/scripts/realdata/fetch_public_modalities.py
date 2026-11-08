#!/usr/bin/env python3
"""Stage B — fetch every public modality source into data/external/<modality>/.

Idempotent + verifiable: each source fetches via its DataSource then is audited
(existence / size / md5). Controlled tiers are rejected at construction by the
DataSource layer and are never referenced here.

Usage:
    python scripts/realdata/fetch_public_modalities.py             # all
    python scripts/realdata/fetch_public_modalities.py string ccle # subset by substring
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mm_tte_survival.data import sources as S  # noqa: E402


def _run(label: str, source, dest: Path):
    t0 = time.time()
    print(f"[fetch] {label} ...", flush=True)
    try:
        source.fetch(dest)
        audit = source.verify()
        dt = time.time() - t0
        status = "ok  " if audit.passed else "WARN"
        print(f"[ {status}] {label}: md5={(audit.checksum or '')[:12]} "
              f"passed={audit.passed} ({dt:.0f}s)  {audit.notes}", flush=True)
        return audit.passed
    except Exception:
        print(f"[FAIL ] {label}:\n{traceback.format_exc()}", flush=True)
        return False


def main(argv):
    filt = [a.lower() for a in argv]
    data = ROOT / "data"
    ok = fail = 0

    for sid, cls in S.PUBLIC_SOURCES.items():
        if filt and not any(f in sid for f in filt):
            continue
        src = cls()
        dest = getattr(src, "dest", None) or (data / "external")
        r = _run(sid, src, Path(dest))
        ok += r; fail += (not r)

    for gse, dest in S.GEO_SERIES.items():
        if filt and not any(f in gse.lower() for f in filt):
            continue
        src = S.GEOSeries(gse, dest=ROOT / dest)
        r = _run(gse, src, ROOT / dest)
        ok += r; fail += (not r)

    print(f"\n[done] ok={ok} fail={fail}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
