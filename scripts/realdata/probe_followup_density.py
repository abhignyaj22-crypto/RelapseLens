#!/usr/bin/env python
"""sota-workaround B0 — feasibility probe: does GDC-open MMRF carry a real longitudinal substrate?

Angle B (un-collapse `follow_ups` -> real landmark-OS, the honest workaround for the BLOCKED dynamic-when) is
only defensible if the serial data actually exists. `fetch_clinical.py` already downloads
`follow_ups.molecular_tests` but collapses each lab to its earliest value (first_lab). This probe counts, from
the REAL GDC fetch, how many MMRF patients have >=3 DATED follow-ups, and how many of those visits carry serial
labs (B2M/albumin/LDH/Hgb/creatinine/calcium). HONEST GATE: unless >=100 patients have >=3 lab-carrying visits,
Angle B stays BLOCKED and is reported as such — no synthetic serial rows are ever written. Real data only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_clinical import fetch_all, LAB_FIELDS  # noqa: E402  (reuse the real GDC fetch)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "autoresearch"; OUT.mkdir(parents=True, exist_ok=True)


def visit_counts(case: dict) -> tuple[int, int]:
    """(# distinct dated follow-ups, # distinct dated follow-ups carrying >=1 real lab)."""
    dated: set[float] = set()
    lab_dated: set[float] = set()
    for fu in case.get("follow_ups") or []:
        day = fu.get("days_to_follow_up")
        if day is None:
            continue
        dated.add(float(day))
        for mt in fu.get("molecular_tests") or []:
            if LAB_FIELDS.get(mt.get("laboratory_test")) is not None and mt.get("test_value") is not None:
                lab_dated.add(float(day))
                break
    return len(dated), len(lab_dated)


def main() -> int:
    cases = fetch_all()
    dated = []
    labbed = []
    for c in cases:
        d, l = visit_counts(c)
        dated.append(d)
        labbed.append(l)
    n = len(cases)

    def ge(counts, k):
        return int(sum(x >= k for x in counts))

    multi = [x for x in dated if x >= 2]
    median_visits_multi = float(sorted(multi)[len(multi) // 2]) if multi else 0.0
    ge3_lab = ge(labbed, 3)
    feasible = ge3_lab >= 100

    report = {
        "cohort": "GDC-open MMRF-COMMPASS (real fetch)", "n_patients": n,
        "dated_followups": {"ge1": ge(dated, 1), "ge2": ge(dated, 2), "ge3": ge(dated, 3),
                            "ge5": ge(dated, 5), "max": int(max(dated) if dated else 0)},
        "lab_carrying_visits": {"ge1": ge(labbed, 1), "ge2": ge(labbed, 2), "ge3": ge3_lab,
                                "ge5": ge(labbed, 5), "max": int(max(labbed) if labbed else 0)},
        "median_dated_visits_among_multivisit": median_visits_multi,
        "frac_multivisit_dated": round(ge(dated, 2) / n, 3) if n else 0.0,
        "gate": "GO — Angle B feasible" if feasible else "BLOCKED — insufficient serial substrate",
        "gate_rule": ">=100 patients with >=3 lab-carrying dated follow-ups",
        "verdict": (f"GO: {ge3_lab} patients have >=3 lab-carrying visits (>=100) -> build real landmark-OS (B1/B2)"
                    if feasible else
                    f"BLOCKED: only {ge3_lab} patients have >=3 lab-carrying visits (<100). Open GDC MMRF is "
                    f"baseline-dominant; a real serial-lab landmark substrate is not available here. Serial "
                    f"CoMMpass is controlled-tier. No synthetic serial data is fabricated — Angle B stays BLOCKED."),
    }
    (OUT / "followup_density_probe.json").write_text(json.dumps(report, indent=2))
    print(f"\n[B0 follow-up density | {report['cohort']}] N={n}")
    print(f"  dated follow-ups:     >=1={report['dated_followups']['ge1']}  >=2={report['dated_followups']['ge2']}"
          f"  >=3={report['dated_followups']['ge3']}  >=5={report['dated_followups']['ge5']}  "
          f"max={report['dated_followups']['max']}")
    print(f"  lab-carrying visits:  >=1={report['lab_carrying_visits']['ge1']}  "
          f">=2={report['lab_carrying_visits']['ge2']}  >=3={ge3_lab}  "
          f">=5={report['lab_carrying_visits']['ge5']}  max={report['lab_carrying_visits']['max']}")
    print(f"  GATE ({report['gate_rule']}): {report['gate']}")
    print(f"  -> {report['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
