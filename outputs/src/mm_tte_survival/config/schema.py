"""Typed config schema (pydantic v2).

`AppConfig` gives main.py clean attribute access (cfg.paths.clinical, cfg.seed)
while staying tolerant of the existing dict-style configs — every block allows
extra keys, and `to_dict()` returns the raw mapping the evaluation/report layers
still consume.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


class Endpoint(_Base):
    name: str = "open_gdc_os"
    claim_scope: str | None = None


class Paths(_Base):
    clinical: str
    cytogenetics: str | None = None
    omics: str | None = None
    program_activity: str | None = None
    outdir: str = "outputs/run"


class Schema(_Base):
    id_col: str = "patient_id"
    time_col: str = "time_months"
    event_col: str = "event"
    split_col: str = "split"
    clinical_cols: list[str] = Field(default_factory=list)
    cytogenetic_cols: list[str] = Field(default_factory=list)


class Cohort(_Base):
    matched_ablation: bool = True
    require_clinical: bool = True
    require_cytogenetics: bool = True
    require_omics: bool = True


class Evaluation(_Base):
    repeated_splits: int = 50
    bootstrap: int = 500
    horizon_months: float = 24.0
    min_test_events: int = 20
    min_subtype_events_hypothesis: int = 10
    min_subtype_events_confirmatory: int = 30


class AppConfig(_Base):
    seed: int = 42
    endpoint: Endpoint = Field(default_factory=Endpoint)
    paths: Paths
    schema_: Schema = Field(default_factory=Schema, alias="schema")
    cohort: Cohort = Field(default_factory=Cohort)
    evaluation: Evaluation = Field(default_factory=Evaluation)

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @property
    def schema(self) -> Schema:  # convenience alias
        return self.schema_

    def to_dict(self) -> dict[str, Any]:
        """Legacy mapping consumed by the evaluation/report layers.

        Normalises the two config dialects: the modern `evaluation:` block is
        also surfaced under the legacy `experiments:`/`validation:` keys so the
        downstream functions work unchanged regardless of which config is used.
        """
        d = self.model_dump(by_alias=True, exclude_none=False)
        ev = d.get("evaluation", {}) or {}
        validation = dict(d.get("validation", {}) or {})
        experiments = dict(d.get("experiments", {}) or {})
        validation.setdefault("repeated_splits", ev.get("repeated_splits", 50))
        validation.setdefault("horizon_months", ev.get("horizon_months", 24.0))
        validation.setdefault("external_validation_available",
                              ev.get("external_validation_available", False))
        for k in ("min_test_events", "min_subtype_events_hypothesis",
                  "min_subtype_events_confirmatory", "bootstrap"):
            if k in ev:
                experiments.setdefault(k, ev[k])
        d["validation"] = validation
        d["experiments"] = experiments
        return d


def load_app_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping")
    return AppConfig.model_validate(raw)
