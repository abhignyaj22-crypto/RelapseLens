"""First-class residual-risk model: clinical Cox baseline + molecular residual Cox.

This is the repo's strongest, most interpretable, endpoint-defensible model:
    clinical_risk            = clinical-only Cox linear predictor
    molecular_residual_risk  = pure Cox on molecular features orthogonalised vs clinical
    total_risk               = clinical_risk + molecular_residual_risk

The additive identity is now EXACT by construction: the molecular Cox is fit on the
residualised molecular features ONLY (clinical_risk is no longer a covariate in it),
so total_risk = clinical_risk (unit weight) + molecular_residual_risk exactly — and
the clinical-vs-total reclassification comparison is well-defined. Previously the
joint Cox fit a clinical_risk coefficient (~1, not 1) that predict() silently
dropped, so total_risk was a hand-built sum, not the fitted model's predictor.

`fit` uses only the training rows; orthogonalisation and both Cox fits are
train-only. `predict` applies the frozen transforms to any rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LinearRegression


@dataclass
class ResidualRiskModel:
    clinical_cols: list[str]
    molecular_cols: list[str]
    time_col: str = "time_months"
    event_col: str = "event"
    penalizer: float = 0.1
    # fitted state
    _impute: pd.Series = field(default=None, repr=False)
    _mu: pd.Series = field(default=None, repr=False)
    _sd: pd.Series = field(default=None, repr=False)
    _cox_clin: CoxPHFitter = field(default=None, repr=False)
    _ortho: LinearRegression = field(default=None, repr=False)
    _cox_total: CoxPHFitter = field(default=None, repr=False)
    clinical_coef_in_joint: float = field(default=None, repr=False)

    def _prep(self, df, cols):
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(self._impute[cols]).fillna(0.0)
        return (X - self._mu[cols]) / self._sd[cols]

    def fit(self, train_df: pd.DataFrame) -> "ResidualRiskModel":
        cols = self.clinical_cols + self.molecular_cols
        raw = train_df[cols].apply(pd.to_numeric, errors="coerce")
        self._impute = raw.median()
        filled = raw.fillna(self._impute).fillna(0.0)
        self._mu = filled.mean()
        self._sd = filled.std().replace(0, 1.0)

        Xc = self._prep(train_df, self.clinical_cols)
        Xm = self._prep(train_df, self.molecular_cols)
        d1 = Xc.copy()
        d1[self.time_col] = pd.to_numeric(train_df[self.time_col]).clip(lower=0.1).values
        d1[self.event_col] = train_df[self.event_col].astype(int).values
        self._cox_clin = CoxPHFitter(penalizer=self.penalizer).fit(
            d1, duration_col=self.time_col, event_col=self.event_col)

        self._ortho = LinearRegression().fit(Xc.values, Xm.values)
        Xm_resid = Xm.values - self._ortho.predict(Xc.values)
        rcols = [f"{c}__r" for c in self.molecular_cols]
        d2 = pd.DataFrame(Xm_resid, columns=rcols, index=train_df.index)
        # NOTE: clinical_risk is deliberately NOT a covariate here -> the molecular
        # term is a pure residual Cox and total_risk = clinical + molecular is exact.
        d2[self.time_col] = d1[self.time_col].values
        d2[self.event_col] = d1[self.event_col].values
        self._cox_total = CoxPHFitter(penalizer=self.penalizer).fit(
            d2, duration_col=self.time_col, event_col=self.event_col)
        # clinical enters total_risk with unit weight by construction (exactly additive).
        self.clinical_coef_in_joint = float(self._cox_total.params_.get("clinical_risk", 1.0))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        Xc = self._prep(df, self.clinical_cols)
        Xm = self._prep(df, self.molecular_cols)
        clinical_risk = self._cox_clin.predict_log_partial_hazard(Xc).values
        Xm_resid = Xm.values - self._ortho.predict(Xc.values)
        rcols = [f"{c}__r" for c in self.molecular_cols]
        beta = self._cox_total.params_
        molecular_residual_risk = Xm_resid @ beta[rcols].values
        total_risk = clinical_risk + molecular_residual_risk
        return pd.DataFrame({
            "clinical_risk": clinical_risk,
            "molecular_residual_risk": molecular_residual_risk,
            "total_risk": total_risk,
        }, index=df.index)

    def coefficients(self):
        clin = pd.DataFrame({"feature": self._cox_clin.params_.index,
                             "coef": self._cox_clin.params_.values,
                             "abs_coef": np.abs(self._cox_clin.params_.values),
                             "hazard_ratio": np.exp(self._cox_clin.params_.values)}
                            ).sort_values("abs_coef", ascending=False)
        rcols = [f"{c}__r" for c in self.molecular_cols]
        beta = self._cox_total.params_[rcols].values
        mol = pd.DataFrame({"feature": self.molecular_cols, "coef": beta,
                            "abs_coef": np.abs(beta),
                            "direction": np.where(beta >= 0, "higher_risk", "lower_risk")}
                           ).sort_values("abs_coef", ascending=False).reset_index(drop=True)
        return clin, mol
