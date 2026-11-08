"""Baseline models for ResistanceMap evaluation.

These baselines are real, runnable models. They accept REAL data from the
caller and raise informative errors if no real data is given. None of them
fabricate data, return zero placeholders, or return random outputs.

Available baselines:

- :class:`ClinicalOnlyBaseline` -- penalized logistic regression on clinical
  features (age, ISS stage, cytogenetic risk).
- :class:`TranscriptomeOnlyBaseline` -- PCA via numpy SVD followed by a
  linear classifier/regressor.
- :class:`LastObservationBaseline` -- per-patient last-observation carry
  forward; requires longitudinal input.
- :class:`SimpleSurvivalBaseline` -- numpy Kaplan-Meier (or Cox if scipy is
  available) stratified by a single covariate.

Functional drug-response baselines (from :mod:`.predictors`, formerly the shadowed top-level
``baselines.py``; A1 1.5 moved it INTO this package so it is reachable): ``predict_global_mean``,
``predict_per_drug_mean``, ``ridge_on_esm2_meanpool``, ``predict_base_rate``, ``logistic_on_esm2``,
``deepcdr_reimpl`` (the DeepCDR stub Phase 2.2 fills in), ``require_baseline_row``.
"""

from __future__ import annotations

from resistancemap.evaluation.baselines.base import Baseline
from resistancemap.evaluation.baselines.clinical_only import ClinicalOnlyBaseline
from resistancemap.evaluation.baselines.transcriptome_only import (
    TranscriptomeOnlyBaseline,
)
from resistancemap.evaluation.baselines.last_observation import LastObservationBaseline
from resistancemap.evaluation.baselines.simple_survival import SimpleSurvivalBaseline
from resistancemap.evaluation.baselines.predictors import (
    predict_global_mean, predict_per_drug_mean, ridge_on_esm2_meanpool,
    predict_base_rate, logistic_on_esm2, deepcdr_reimpl, require_baseline_row,
    BASELINES_REGRESSION, BASELINES_CLASSIFICATION,
)

__all__ = [
    "Baseline",
    "ClinicalOnlyBaseline",
    "TranscriptomeOnlyBaseline",
    "LastObservationBaseline",
    "SimpleSurvivalBaseline",
    # functional drug-response baselines (predictors.py)
    "predict_global_mean", "predict_per_drug_mean", "ridge_on_esm2_meanpool",
    "predict_base_rate", "logistic_on_esm2", "deepcdr_reimpl", "require_baseline_row",
    "BASELINES_REGRESSION", "BASELINES_CLASSIFICATION",
]
