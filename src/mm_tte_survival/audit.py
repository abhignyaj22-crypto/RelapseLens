from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def audit_inputs(clinical_path: str, cytogenetics_path: str | None, omics_path: str | None, out: str | Path,
                 id_col: str = "patient_id", time_col: str = "time_months", event_col: str = "event") -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"tables": {}, "gates": {}, "usable_for": {}}
    clinical = pd.read_csv(clinical_path)
    report["tables"]["clinical"] = {
        "path": clinical_path,
        "n_rows": int(len(clinical)),
        "n_patients": int(clinical[id_col].nunique()) if id_col in clinical.columns else 0,
        "columns": list(clinical.columns),
        "duplicate_patient_rows": int(clinical[id_col].duplicated().sum()) if id_col in clinical.columns else None,
    }
    has_survival = {id_col, time_col, event_col}.issubset(clinical.columns)
    if has_survival:
        event_n = int(pd.to_numeric(clinical[event_col], errors="coerce").fillna(0).sum())
        report["tables"]["clinical"].update({
            "events": event_n,
            "median_time": float(pd.to_numeric(clinical[time_col], errors="coerce").median()),
            "missing_time": int(pd.to_numeric(clinical[time_col], errors="coerce").isna().sum()),
            "missing_event": int(pd.to_numeric(clinical[event_col], errors="coerce").isna().sum()),
        })
    report["usable_for"]["survival_tte"] = bool(has_survival and report["tables"]["clinical"].get("n_patients", 0) >= 50 and report["tables"]["clinical"].get("events", 0) >= 20)

    subtype_cols = []
    if cytogenetics_path and Path(cytogenetics_path).exists():
        cyto = pd.read_csv(cytogenetics_path)
        subtype_cols = [c for c in cyto.columns if c != id_col and pd.api.types.is_numeric_dtype(cyto[c])]
        overlap = len(set(clinical[id_col].astype(str)) & set(cyto[id_col].astype(str))) if id_col in cyto.columns and id_col in clinical.columns else 0
        report["tables"]["cytogenetics"] = {
            "path": cytogenetics_path,
            "n_rows": int(len(cyto)),
            "n_patients": int(cyto[id_col].nunique()) if id_col in cyto.columns else 0,
            "patient_overlap_with_clinical": int(overlap),
            "subtype_cols": subtype_cols,
        }
        if has_survival:
            merged = clinical[[id_col, time_col, event_col]].merge(cyto, on=id_col, how="inner")
            subtype_counts = {}
            for c in subtype_cols:
                m = pd.to_numeric(merged[c], errors="coerce").fillna(0) > 0
                subtype_counts[c] = {"n": int(m.sum()), "events": int(merged.loc[m, event_col].sum())}
            report["tables"]["cytogenetics"]["subtype_counts"] = subtype_counts
    report["usable_for"]["cytogenetic_subtype_tte"] = bool(subtype_cols and report["usable_for"].get("survival_tte"))

    if omics_path and Path(omics_path).exists():
        omics = pd.read_csv(omics_path, nrows=5)
        full_index = pd.read_csv(omics_path, usecols=[id_col]) if id_col in omics.columns else pd.DataFrame()
        n_numeric = sum(1 for c in omics.columns if c != id_col and pd.api.types.is_numeric_dtype(omics[c]))
        overlap = len(set(clinical[id_col].astype(str)) & set(full_index[id_col].astype(str))) if not full_index.empty and id_col in clinical.columns else 0
        report["tables"]["omics"] = {
            "path": omics_path,
            "n_rows": int(len(full_index)) if not full_index.empty else None,
            "patient_overlap_with_clinical": int(overlap),
            "n_numeric_feature_columns_sampled": int(n_numeric),
            "note": "Omics is model-usable only if patient_id joins to survival labels and transformations are train-split fitted only.",
        }
        report["usable_for"]["multiomic_tte"] = bool(report["usable_for"].get("survival_tte") and overlap >= 50 and n_numeric >= 5)
    else:
        report["usable_for"]["multiomic_tte"] = False

    n = report["tables"]["clinical"].get("n_patients", 0)
    events = report["tables"]["clinical"].get("events", 0)
    report["gates"] = {
        "min_patients_50": n >= 50,
        "min_events_20": events >= 20 if events is not None else False,
        "one_row_per_patient_required": report["tables"]["clinical"].get("duplicate_patient_rows", 1) == 0,
        "publication_grade_subtype_modeling": bool(n >= 200 and events >= 60 and subtype_cols),
        "longitudinal_first_hitting_claim": "blocked unless repeated time-stamped molecular/clinical states exist before event/censoring",
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def write_markdown_audit(report: dict, out_md: str | Path) -> None:
    out_md = Path(out_md)
    lines = ["# Data Usability Audit", "", "## Gate summary"]
    for k, v in report.get("gates", {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Usable for")
    for k, v in report.get("usable_for", {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Tables")
    for name, spec in report.get("tables", {}).items():
        lines.append(f"### {name}")
        for k, v in spec.items():
            if k == "columns":
                lines.append(f"- {k}: {len(v)} columns")
            else:
                lines.append(f"- {k}: {v}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
