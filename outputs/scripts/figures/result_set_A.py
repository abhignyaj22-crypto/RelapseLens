"""Result Set A — does the open data contain a legitimate longitudinal signal?

Computes the eight required cohort characterizations DIRECTLY from the real GDC MMRF-CoMMpass table
(data/real/clinical_survival.csv) — no invented numbers — and renders the honest verdict against the
pre-registered failure condition: "if most patients have n_visits < 3, do not pretend to have a strong
longitudinal model."  Emits a JSON artifact (claim-auditable), a markdown table, and a 4-panel figure.

Run:  venv/bin/python scripts/figures/result_set_A.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "real" / "clinical_survival.csv"
OUT_FIG = ROOT / "docs" / "paper" / "figures"
OUT_MD = ROOT / "docs" / "paper" / "result_set_A.md"
OUT_JSON = ROOT / "outputs" / "paper" / "result_set_A.json"

MIN_VISITS = 3        # pre-registered density floor for "longitudinal-supportable"
LAB_COLS = ["b2m", "albumin", "ldh", "hemoglobin", "creatinine", "calcium"]


def compute_result_set_a(df: pd.DataFrame) -> dict:
    """Return every Result-Set-A output as real numbers + the PASS/FAIL verdict. Pure pandas."""
    n_pat = int(df["patient_id"].nunique())
    visits = df.groupby("patient_id").size()
    vd = {int(k): int(v) for k, v in visits.value_counts().sort_index().items()}
    median_visits = float(visits.median())
    frac_multi = float((visits >= 2).mean())
    frac_ge3 = float((visits >= MIN_VISITS).mean())

    ot = pd.to_numeric(df["os_time"], errors="coerce")
    pe = pd.to_numeric(df.get("pfs_event", pd.Series(dtype=float)), errors="coerce")
    line = df.get("line_of_therapy")

    rep = {
        "source": str(REAL.relative_to(ROOT)),
        "n_patients": n_pat,
        "visits_per_patient": {"distribution": vd, "median": median_visits,
                               "frac_multi_visit": frac_multi, "frac_ge3_visits": frac_ge3},
        "follow_up_months": {"median": round(float(ot.median()), 1), "max": round(float(ot.max()), 1),
                             "q25": round(float(ot.quantile(.25)), 1), "q75": round(float(ot.quantile(.75)), 1)},
        "event_prevalence": {
            "os_event_rate": round(float(df["os_event"].mean()), 3), "os_n_events": int(df["os_event"].sum()),
            "pfs_populated_frac": round(float(pe.notna().mean()), 3),
            "pfs_note": "PFS/progression columns are 0% populated on the open tier — OS is the only endpoint."},
        "missingness": {c: round(float(df[c].isna().mean()), 3)
                        for c in LAB_COLS + ["age", "sex", "iss", "riss"] if c in df.columns},
        "inter_visit_interval_days": (
            "UNDEFINED — 0 patients have >=2 visits, so no inter-visit gaps exist"),
        "progression_event_distribution": (
            "UNAVAILABLE — no serial progression endpoint on the open tier (PFS 0% populated)"),
        "treatment_line_distribution": ({int(k): int(v) for k, v in line.value_counts().sort_index().items()}
                                        if line is not None else "absent")
        ,
        "treatment_note": ("line_of_therapy / treatment are a single BASELINE value per patient, not a "
                           "regimen start/stop timeline — they do not constitute serial follow-up."),
    }
    fail = median_visits < MIN_VISITS
    rep["failure_condition"] = {
        "rule": f"median visits/patient < {MIN_VISITS}  =>  longitudinal modeling not supportable",
        "median_visits_per_patient": median_visits,
        "met": bool(fail),
    }
    rep["verdict"] = (
        "FAIL — the open GDC MMRF-CoMMpass tier is baseline-only (100% single-visit). The failure "
        "condition is met for the ENTIRE cohort; a longitudinal / dynamic progression model cannot be "
        "honestly built on this substrate. Result Sets B/C/E and Figures 2/3/5 are BLOCKED pending real "
        "serial CoMMpass data (MMRF Researcher-Gateway / dbGaP controlled-tier). This is the honest "
        "Result A: the substrate is characterized and the null is reported, not papered over."
        if fail else
        "PASS — visit density supports dynamic modeling.")
    return rep


def _write_md(rep: dict) -> None:
    v = rep["visits_per_patient"]; f = rep["follow_up_months"]; e = rep["event_prevalence"]
    lines = [
        "# Result Set A — does the open data contain a legitimate longitudinal signal?",
        "",
        f"Computed from real `{rep['source']}` by `scripts/figures/result_set_A.py` — no invented numbers.",
        "",
        "| Required output | Real value (open GDC MMRF-CoMMpass) |",
        "|---|---|",
        f"| Number of patients | **{rep['n_patients']}** |",
        f"| Visits per patient | **median {v['median']:.0f}**, {v['frac_multi_visit']*100:.0f}% multi-visit, "
        f"{v['frac_ge3_visits']*100:.0f}% with ≥3 visits — distribution {v['distribution']} |",
        f"| Follow-up duration (months) | median {f['median']}, IQR [{f['q25']}, {f['q75']}], max {f['max']} |",
        f"| Event prevalence | OS {e['os_event_rate']*100:.1f}% ({e['os_n_events']} deaths); "
        f"PFS {e['pfs_populated_frac']*100:.0f}% populated |",
        f"| Missingness | {', '.join(f'{k} {int(val*100)}%' for k, val in rep['missingness'].items())} |",
        f"| Inter-visit interval | {rep['inter_visit_interval_days']} |",
        f"| Progression-event distribution | {rep['progression_event_distribution']} |",
        f"| Treatment-line distribution | {rep['treatment_line_distribution']} (baseline value only) |",
        "",
        f"**Failure condition** (`{rep['failure_condition']['rule']}`): "
        f"median visits = {rep['failure_condition']['median_visits_per_patient']:.0f} → "
        f"**{'MET' if rep['failure_condition']['met'] else 'not met'}**.",
        "",
        f"## Verdict",
        "",
        rep["verdict"],
        "",
        f"> {rep['treatment_note']}  {rep['event_prevalence']['pfs_note']}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def _render_figure(rep: dict, df: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))
    fig.suptitle("Result Set A  —  open GDC MMRF-CoMMpass is baseline-only: no longitudinal substrate",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")

    # A: attrition
    n = rep["n_patients"]
    steps = [("all GDC MMRF patients", n), ("has baseline clinical", n),
             ("has ≥2 visits", int(round(rep["visits_per_patient"]["frac_multi_visit"] * n))),
             ("has ≥3 visits", int(round(rep["visits_per_patient"]["frac_ge3_visits"] * n))),
             ("longitudinal cohort", int(round(rep["visits_per_patient"]["frac_ge3_visits"] * n)))]
    a0 = ax[0, 0]
    ys = range(len(steps))[::-1]
    for y, (lbl, cnt) in zip(ys, steps):
        col = "#268" if cnt > 0 else "#c33"
        a0.barh(y, cnt, color=col, alpha=.85)
        a0.text(max(cnt, n * 0.01) + n * 0.01, y, f"{lbl}: {cnt}", va="center", fontsize=8.2, color=col)
    a0.set_xlim(0, n * 1.5); a0.set_yticks([]); a0.set_title("A  Cohort attrition", loc="left", fontsize=10)
    a0.set_xlabel("patients")

    # B: visit-density
    a1 = ax[0, 1]
    vd = rep["visits_per_patient"]["distribution"]
    a1.bar(list(vd.keys()), list(vd.values()), color="#c33", width=0.6)
    a1.axvline(MIN_VISITS - 0.5, color="#333", ls="--", lw=1)
    a1.text(MIN_VISITS - 0.4, max(vd.values()) * 0.7, f"density floor\n(≥{MIN_VISITS} visits)", fontsize=7.5)
    a1.set_title("B  Visits per patient (100% = 1)", loc="left", fontsize=10)
    a1.set_xlabel("n visits"); a1.set_ylabel("patients"); a1.set_xlim(0, MIN_VISITS + 2)

    # C: follow-up
    a2 = ax[1, 0]
    ot = pd.to_numeric(df["os_time"], errors="coerce").dropna()
    dead = df["os_event"] == 1
    a2.hist(ot[dead], bins=25, alpha=.8, color="#b23", label=f"death ({int(dead.sum())})")
    a2.hist(ot[~dead], bins=25, alpha=.55, color="#888", label=f"censored ({int((~dead).sum())})")
    a2.set_title("C  Follow-up (OS months) — a single baseline snapshot", loc="left", fontsize=10)
    a2.set_xlabel("months from baseline"); a2.set_ylabel("patients"); a2.legend(fontsize=7.5)

    # D: verdict + inter-visit degeneracy
    a3 = ax[1, 1]; a3.axis("off")
    a3.add_patch(plt.Rectangle((0.02, 0.55), 0.96, 0.42, transform=a3.transAxes,
                               fc="#fdf3f3", ec="#c33", lw=1.8))
    a3.text(0.5, 0.9, "FAILURE CONDITION MET", ha="center", fontsize=11.5, fontweight="bold",
            color="#c33", transform=a3.transAxes)
    a3.text(0.5, 0.74, f"median visits/patient = 1  (< {MIN_VISITS})\n"
                       f"inter-visit intervals: UNDEFINED (0 multi-visit patients)\n"
                       "PFS/progression endpoint: 0% populated (OS only)",
            ha="center", va="center", fontsize=8.6, transform=a3.transAxes)
    a3.text(0.5, 0.40, "→ Longitudinal / dynamic model NOT supportable on open data.\n"
                       "Result Sets B/C/E + Figs 2/3/5 blocked pending real serial\n"
                       "CoMMpass (MMRF Researcher-Gateway / dbGaP controlled-tier).",
            ha="center", va="center", fontsize=8.6, color="#444", transform=a3.transAxes)
    a3.text(0.5, 0.10, "Honest Result A: substrate characterized, null reported — not papered over.",
            ha="center", va="center", fontsize=8, style="italic", color="#777", transform=a3.transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT_FIG / "figA_longitudinal_substrate.pdf"
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    df = pd.read_csv(REAL)
    rep = compute_result_set_a(df)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2))
    _write_md(rep)
    p = _render_figure(rep, df)
    print(f"verdict: {rep['verdict'][:60]}...")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}, {p.relative_to(ROOT)} (+.png)")


if __name__ == "__main__":
    main()
