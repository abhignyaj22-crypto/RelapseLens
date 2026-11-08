#!/usr/bin/env python
"""Step 1: build clinical_survival.csv from real GDC MMRF-COMMPASS clinical data.

Source: GDC open-access clinical API (api.gdc.cancer.gov), project MMRF-COMMPASS.
Endpoint used here is the public OS endpoint: vital_status + days_to_death /
days_to_last_follow_up. PFS/progression dates are NOT distributed in GDC open
clinical, so pfs_time/pfs_event are emitted empty and must be sourced from the
MMRF Researcher Gateway (controlled) if needed. No synthetic values are written.
"""
from __future__ import annotations
import json, urllib.parse, urllib.request, csv, sys
from pathlib import Path

API = "https://api.gdc.cancer.gov/cases"
OUT = Path(__file__).resolve().parents[2] / "data" / "real" / "clinical_survival.csv"
DAYS_PER_MONTH = 30.4375

FILT = {"op": "in", "content": {"field": "project.project_id", "value": ["MMRF-COMMPASS"]}}
EXPAND = "demographic,diagnoses,diagnoses.treatments,follow_ups,follow_ups.molecular_tests"

# Baseline labs (earliest available value per patient) we lift out of molecular_tests.
LAB_FIELDS = {
    "Beta 2 Microglobulin": "b2m",
    "Albumin": "albumin",
    "Lactate Dehydrogenase": "ldh",
    "Hemoglobin": "hemoglobin",
    "Creatinine": "creatinine",
    "Calcium": "calcium",
}
ISS_MAP = {"I": 1, "II": 2, "III": 3}


def fetch_all() -> list[dict]:
    hits, frm, size = [], 0, 100
    while True:
        params = {
            "filters": json.dumps(FILT), "expand": EXPAND,
            "format": "JSON", "size": str(size), "from": str(frm),
        }
        url = API + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=180) as r:
            d = json.load(r)
        page = d["data"]["hits"]
        hits.extend(page)
        total = d["data"]["pagination"]["total"]
        frm += size
        print(f"  fetched {len(hits)}/{total}", file=sys.stderr)
        if frm >= total:
            break
    return hits


def first_lab(case: dict) -> dict:
    """Earliest (lowest days_to_follow_up) value per lab type."""
    best: dict[str, tuple[float, float]] = {}
    for fu in case.get("follow_ups") or []:
        day = fu.get("days_to_follow_up")
        day = float(day) if day is not None else 1e9
        for mt in fu.get("molecular_tests") or []:
            lab = LAB_FIELDS.get(mt.get("laboratory_test"))
            val = mt.get("test_value")
            if lab is None or val is None:
                continue
            if lab not in best or day < best[lab][0]:
                best[lab] = (day, float(val))
    return {k: v[1] for k, v in best.items()}


def os_time_event(case: dict) -> tuple[float | None, int]:
    demo = case.get("demographic") or {}
    vital = (demo.get("vital_status") or "").lower()
    if vital == "dead":
        d = demo.get("days_to_death")
        if d is not None:
            return float(d), 1
    # censored: take the largest follow-up day available
    cand = []
    for dx in case.get("diagnoses") or []:
        for f in ("days_to_last_follow_up", "days_to_last_known_disease_status"):
            if dx.get(f) is not None:
                cand.append(float(dx[f]))
    for fu in case.get("follow_ups") or []:
        if fu.get("days_to_follow_up") is not None:
            cand.append(float(fu["days_to_follow_up"]))
    if vital == "dead":  # Dead but no days_to_death -> use last known time as event time
        if cand:
            return max(cand), 1
        return None, 1
    if cand:
        return max(cand), 0
    return None, 0


def n_lines(case: dict) -> int:
    lines = set()
    for dx in case.get("diagnoses") or []:
        for tx in dx.get("treatments") or []:
            lot = tx.get("regimen_or_line_of_therapy")
            if lot:
                lines.add(lot)
    return len(lines)


def main():
    cases = fetch_all()
    rows, skipped = [], 0
    for c in cases:
        demo = c.get("demographic") or {}
        dxs = c.get("diagnoses") or []
        t_days, event = os_time_event(c)
        if t_days is None or t_days <= 0:
            skipped += 1
            continue
        os_months = round(t_days / DAYS_PER_MONTH, 4)
        age = demo.get("age_at_index")
        if age is None and dxs and dxs[0].get("age_at_diagnosis") is not None:
            age = round(dxs[0]["age_at_diagnosis"] / 365.25, 1)
        sex = (demo.get("sex_at_birth") or demo.get("gender") or "").lower()
        sex = "M" if sex.startswith("m") else ("F" if sex.startswith("f") else "")
        iss_raw = next((dx.get("iss_stage") for dx in dxs if dx.get("iss_stage")), None)
        iss = ISS_MAP.get(iss_raw, "")
        lines = n_lines(c)
        labs = first_lab(c)
        row = {
            "patient_id": c.get("submitter_id"),
            "time": os_months,           # primary endpoint = OS (months)
            "event": event,
            "age": age if age is not None else "",
            "sex": sex,
            "iss": iss,
            "riss": "",                  # needs LDH-ULN + FISH translocations; not reliably in GDC open
            "treatment": lines,          # number of distinct lines of therapy on record
            "os_time": os_months,
            "os_event": event,
            "pfs_time": "",              # not distributed in GDC open clinical
            "pfs_event": "",
            # ---- model-ready encodings + real baseline labs (extra columns) ----
            "time_months": os_months,
            "sex_M": 1 if sex == "M" else (0 if sex == "F" else ""),
            "iss_2": 1 if iss == 2 else (0 if iss in (1, 3) else ""),
            "iss_3": 1 if iss == 3 else (0 if iss in (1, 2) else ""),
            "line_of_therapy": lines,
            "b2m": labs.get("b2m", ""),
            "albumin": labs.get("albumin", ""),
            "ldh": labs.get("ldh", ""),
            "hemoglobin": labs.get("hemoglobin", ""),
            "creatinine": labs.get("creatinine", ""),
            "calcium": labs.get("calcium", ""),
        }
        rows.append(row)
    cols = list(rows[0].keys())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    n_event = sum(r["event"] for r in rows)
    print(f"wrote {OUT}  N={len(rows)} patients, deaths(events)={n_event}, skipped(no time)={skipped}")


if __name__ == "__main__":
    main()
