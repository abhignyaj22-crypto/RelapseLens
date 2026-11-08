#!/usr/bin/env python
"""Within-clinical-stratum reclassification analysis (corrects the misleading
"HR vs rest" framing).

"HR vs rest" is confounded: clinical-low/molecular-high vs ALL-rest looks
protective only because the rest includes clinical-high patients. The scientific
question is WITHIN a clinical-risk stratum: does molecular-high vs molecular-low
separate outcomes?

Models (binary clinical/molecular at the median of clinical_risk / molecular_residual_risk):
  1. OS ~ clinical_bin + molecular_bin + clinical_bin:molecular_bin   (interaction)
  2. within clinical-low : OS ~ molecular_high
  3. within clinical-high: OS ~ molecular_high

Outputs:
  reclassification_within_stratum_hr.csv
  reclassification_within_stratum_logrank.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "experiment0_open_gdc_os"
DEC = D / "residual_risk_decomposition.csv"


def main():
    dec = pd.read_csv(DEC)
    tcol = "time_months" if "time_months" in dec.columns else dec.columns[2]
    ecol = "event"
    t = pd.to_numeric(dec[tcol]).values
    e = dec[ecol].astype(int).values
    # binary split at median (rank-based, ties-robust)
    clin_hi = (pd.Series(dec.clinical_risk.values).rank(pct=True) > 0.5).values
    mol_hi = (pd.Series(dec.molecular_residual_risk.values).rank(pct=True) > 0.5).values

    hr_rows, lr_rows = [], []

    # 1. interaction model
    df1 = pd.DataFrame({"t": t, "e": e, "clinical_high": clin_hi.astype(int),
                        "molecular_high": mol_hi.astype(int)})
    df1["clin_x_mol"] = df1["clinical_high"] * df1["molecular_high"]
    cph = CoxPHFitter().fit(df1, duration_col="t", event_col="e")
    for term in ["clinical_high", "molecular_high", "clin_x_mol"]:
        hr_rows.append({"analysis": "interaction_model", "stratum": "all", "term": term,
                        "HR": round(float(np.exp(cph.params_[term])), 3),
                        "ci_low": round(float(np.exp(cph.confidence_intervals_.loc[term].iloc[0])), 3),
                        "ci_high": round(float(np.exp(cph.confidence_intervals_.loc[term].iloc[1])), 3),
                        "p": round(float(cph.summary.loc[term, "p"]), 4)})

    # 2 & 3. within-stratum molecular_high effect
    for stratum_name, smask in [("clinical_low", ~clin_hi), ("clinical_high", clin_hi)]:
        ts, es, ms = t[smask], e[smask], mol_hi[smask]
        n, ev = int(smask.sum()), int(es.sum())
        # Cox within stratum
        try:
            d = pd.DataFrame({"t": ts, "e": es, "molecular_high": ms.astype(int)})
            c = CoxPHFitter().fit(d, duration_col="t", event_col="e")
            hr = float(np.exp(c.params_["molecular_high"]))
            lo = float(np.exp(c.confidence_intervals_.loc["molecular_high"].iloc[0]))
            hi = float(np.exp(c.confidence_intervals_.loc["molecular_high"].iloc[1]))
            p = float(c.summary.loc["molecular_high", "p"])
        except Exception as ex:
            hr = lo = hi = p = np.nan
        hr_rows.append({"analysis": "within_stratum", "stratum": stratum_name,
                        "term": "molecular_high_vs_low", "HR": round(hr, 3),
                        "ci_low": round(lo, 3), "ci_high": round(hi, 3), "p": round(p, 4),
                        "n": n, "events": ev})
        # log-rank within stratum: molecular high vs low
        lr = logrank_test(ts[ms], ts[~ms], es[ms], es[~ms])
        lr_rows.append({"stratum": stratum_name, "contrast": "molecular_high_vs_low",
                        "n_high": int(ms.sum()), "n_low": int((~ms).sum()),
                        "events_high": int(es[ms].sum()), "events_low": int(es[~ms].sum()),
                        "logrank_p": round(float(lr.p_value), 4)})

    pd.DataFrame(hr_rows).to_csv(D / "reclassification_within_stratum_hr.csv", index=False)
    pd.DataFrame(lr_rows).to_csv(D / "reclassification_within_stratum_logrank.csv", index=False)
    print("=== within-stratum HR ===")
    print(pd.DataFrame(hr_rows).to_string(index=False))
    print("\n=== within-stratum log-rank ===")
    print(pd.DataFrame(lr_rows).to_string(index=False))


if __name__ == "__main__":
    main()
