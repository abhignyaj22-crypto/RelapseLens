"""Data contracts & validation checks (pydantic-backed).

Validates the modality tables and the endpoint/claim context BEFORE modeling and
writes validation_report.json + data_contract_report.md. Failures are recorded
(not silently ignored); hard failures raise.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .loaders import RawModalities


def _check(name, ok, detail=""):
    return {"check": name, "passed": bool(ok), "detail": detail}


def validate_all_inputs(raw: RawModalities, endpoint: dict, schema: dict,
                        output_dir: str | Path) -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    id_col = schema.get("id_col", "patient_id")
    time_col = schema.get("time_col", "time_months")
    event_col = schema.get("event_col", "event")
    clin = raw.clinical
    checks = []

    checks.append(_check("clinical_present", clin is not None))
    checks.append(_check("patient_id_present", id_col in clin.columns))
    checks.append(_check("patient_id_unique", not clin[id_col].duplicated().any(),
                         f"{int(clin[id_col].duplicated().sum())} duplicates"))
    checks.append(_check("patient_id_nonnull", clin[id_col].notna().all()))
    t = pd.to_numeric(clin.get(time_col), errors="coerce")
    checks.append(_check("time_positive", bool((t > 0).all()),
                         f"{int((t <= 0).sum())} non-positive / NaN times"))
    e = pd.to_numeric(clin.get(event_col), errors="coerce")
    checks.append(_check("event_binary", bool(set(pd.unique(e.dropna())) <= {0, 1}),
                         f"values={sorted(set(pd.unique(e.dropna())))[:5]}"))
    checks.append(_check("endpoint_declared", bool(endpoint.get("name"))))
    checks.append(_check("endpoint_type_valid",
                         endpoint.get("endpoint_type") in {
                             "overall_survival", "progression_free_survival",
                             "ttnt_proxy", "early_progression_landmark"},
                         endpoint.get("endpoint_type", "")))

    # provenance present for cytogenetic features
    prov_ok = raw.provenance is not None and "feature" in (raw.provenance.columns if raw.provenance is not None else [])
    checks.append(_check("cytogenetics_provenance_present", prov_ok))

    # omics numeric
    if raw.omics is not None:
        num = all(pd.api.types.is_numeric_dtype(raw.omics[c]) for c in raw.omics.columns if c != id_col)
        checks.append(_check("omics_features_numeric", num))

    # no outcome columns leak into a feature table
    leak = []
    for tbl_name, tbl in [("cytogenetics", raw.cytogenetics), ("omics", raw.omics),
                          ("program_activity", raw.program_activity)]:
        if tbl is not None and (time_col in tbl.columns or event_col in tbl.columns):
            leak.append(tbl_name)
    checks.append(_check("no_outcome_leak_into_features", not leak, f"leaking: {leak}"))

    report = {
        "endpoint": endpoint.get("name"),
        "endpoint_type": endpoint.get("endpoint_type"),
        "n_patients_clinical": int(len(clin)),
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }
    (out / "validation_report.json").write_text(json.dumps(report, indent=2))
    _write_md(report, out / "data_contract_report.md")

    hard = [c for c in checks if not c["passed"] and c["check"] in {
        "clinical_present", "patient_id_present", "patient_id_unique",
        "time_positive", "event_binary", "no_outcome_leak_into_features"}]
    if hard:
        raise ValueError(f"Data contract hard failures: {[c['check'] for c in hard]}")
    return report


def _write_md(report, path):
    L = ["# Data contract report", "",
         f"Endpoint: `{report['endpoint']}` ({report['endpoint_type']}) · "
         f"N={report['n_patients_clinical']} · all_passed={report['all_passed']}", "",
         "| Check | Passed | Detail |", "|---|---|---|"]
    for c in report["checks"]:
        L.append(f"| {c['check']} | {'✅' if c['passed'] else '❌'} | {c['detail']} |")
    Path(path).write_text("\n".join(L) + "\n")
