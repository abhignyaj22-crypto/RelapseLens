"""Leakage detection utilities for evaluation governance.

Audits that live here:

* :func:`audit_patient_split` -- structural overlap of patient IDs across
  train / val / test splits.
* :func:`detect_temporal_leakage` -- per-patient time ordering across splits,
  i.e. ``max(train_time(pid)) >= min(test_time(pid))``.
* :func:`detect_specimen_alias_leakage` -- same biological specimen under
  different IDs across splits (e.g. DepMap ACH-* vs CCLE names).
* :func:`detect_future_covariate_leakage` -- features measured after the
  prediction time.
* :func:`detect_drug_exposure_leakage` -- drug exposure after prediction
  time leaking into input features.
* :func:`detect_pseudotime_calendar_confusion` -- pseudotime values used
  where calendar time is required.
* :func:`detect_external_cohort_overlap` -- external validation cohort
  overlaps with pretraining data.
* :func:`run_full_leakage_audit` -- runs ALL checks, returns comprehensive
  report with PASS/FAIL per check.
* :func:`freeze_split_manifest` -- writes ``split_manifest.json`` with
  SHA-256 hashes of each split for reproducibility.

All functions are pure-python and treat numpy as optional. They never
fabricate data: if the inputs are empty or the timestamp field is missing,
they record that explicitly in the returned :class:`LeakageReport`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass
class LeakageReport:
    """Result of a leakage audit.

    Attributes:
        duplicate_patient_ids: Patient IDs that appear in more than one
            split. The dict maps ``patient_id -> [split_names]``.
        temporal_leaks: List of dicts describing patients whose train-time
            is greater than or equal to their test-time. Each entry has
            ``patient_id``, ``train_time``, ``test_time``, ``split_pair``.
        clean: ``True`` iff there were no duplicates **and** no temporal
            leaks **and** the audit had something to audit.
        details: Free-form bag for additional diagnostics
            (e.g. ``audited_n``, ``timestamp_field``, ``warnings``).
    """

    duplicate_patient_ids: dict[str, list[str]] = field(default_factory=dict)
    temporal_leaks: list[dict[str, Any]] = field(default_factory=list)
    clean: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def overlaps(self) -> list[str]:
        """Patient IDs that appear in more than one split."""
        return list(self.duplicate_patient_ids.keys())

    @property
    def violations(self) -> list[dict[str, Any]]:
        """Temporal leakage violations (alias for :attr:`temporal_leaks`)."""
        return list(self.temporal_leaks)

    @property
    def leaking_patients(self) -> list[str]:
        """Patient IDs flagged for temporal leakage."""
        return [v.get("patient_id") for v in self.temporal_leaks if "patient_id" in v]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view of the report."""
        return {
            "duplicate_patient_ids": dict(self.duplicate_patient_ids),
            "temporal_leaks": [dict(t) for t in self.temporal_leaks],
            "clean": self.clean,
            "details": dict(self.details),
        }


def _to_id_list(ids: Iterable[Any]) -> list[str]:
    """Coerce an iterable of IDs into a list of strings (preserving order)."""
    out: list[str] = []
    for item in ids:
        if item is None:
            continue
        out.append(str(item))
    return out


def audit_patient_split(
    train_ids: Iterable[Any] | None = None,
    val_ids: Iterable[Any] | None = None,
    test_ids: Iterable[Any] | None = None,
    *,
    train: Iterable[Any] | None = None,
    val: Iterable[Any] | None = None,
    test: Iterable[Any] | None = None,
) -> LeakageReport:
    """Detect duplicate patient IDs across train / val / test splits.

    Args:
        train_ids / train: Iterable of patient IDs assigned to the training
            split. Either name is accepted; ``train`` is the canonical
            short form for callers that prefer keyword arguments.
        val_ids / val: Iterable of patient IDs assigned to the validation
            split.
        test_ids / test: Iterable of patient IDs assigned to the test split.

    Returns:
        A :class:`LeakageReport`. ``clean`` is ``True`` only when none of
        the inputs were empty and there were zero overlaps.
    """
    train_arg = train_ids if train_ids is not None else (train if train is not None else [])
    val_arg = val_ids if val_ids is not None else (val if val is not None else [])
    test_arg = test_ids if test_ids is not None else (test if test is not None else [])
    train = _to_id_list(train_arg)
    val = _to_id_list(val_arg)
    test = _to_id_list(test_arg)

    membership: dict[str, list[str]] = defaultdict(list)
    for pid in train:
        membership[pid].append("train")
    for pid in val:
        membership[pid].append("val")
    for pid in test:
        membership[pid].append("test")

    duplicates = {
        pid: splits for pid, splits in membership.items() if len(set(splits)) > 1
    }

    details: dict[str, Any] = {
        "n_train": len(set(train)),
        "n_val": len(set(val)),
        "n_test": len(set(test)),
        "n_unique_total": len(membership),
    }
    warnings: list[str] = []
    if not train or not test:
        warnings.append("train or test split is empty; audit is vacuous")
    if not val:
        warnings.append("validation split is empty")
    if warnings:
        details["warnings"] = warnings

    clean = not duplicates and bool(train) and bool(test)
    if duplicates:
        logger.error(
            "audit_patient_split: %d patient(s) overlap across splits",
            len(duplicates),
        )
    return LeakageReport(
        duplicate_patient_ids=duplicates,
        temporal_leaks=[],
        clean=clean,
        details=details,
    )


def _coerce_timestamp(value: Any) -> float | None:
    """Best-effort conversion of an arbitrary timestamp into a float.

    Accepts ``datetime``, ``date``, ISO-format strings, or numeric values.
    Returns ``None`` if the value cannot be parsed (the caller decides what
    to do; this function never raises on bad input).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def detect_temporal_leakage(
    samples: Sequence[Mapping[str, Any]] | None = None,
    split_assignments: Mapping[Any, str] | None = None,
    timestamp_field: str = "collection_date",
    patient_id_field: str = "patient_id",
    *,
    train: Sequence[Mapping[str, Any]] | None = None,
    val: Sequence[Mapping[str, Any]] | None = None,
    test: Sequence[Mapping[str, Any]] | None = None,
) -> LeakageReport:
    """Flag temporal leakage where train samples are not strictly before test.

    For every patient that appears in more than one split, the function
    compares ``max(train_time)`` against ``min(test_time)`` (and val against
    test, and train against val). If the train-side max is greater than or
    equal to the test-side min, the patient is flagged.

    Args:
        samples: Iterable of sample dicts. Each dict must carry at least
            ``patient_id_field`` and ``timestamp_field``.
        split_assignments: Mapping ``sample_index -> split_name`` (split
            name in ``{"train", "val", "test"}``). Sample index matches the
            position in ``samples``.
        timestamp_field: Key inside each sample dict carrying the
            collection timestamp. Defaults to ``"collection_date"``.
        patient_id_field: Key inside each sample dict carrying the patient
            identifier. Defaults to ``"patient_id"``.

    Returns:
        A :class:`LeakageReport`. ``clean`` is ``True`` iff no leaks were
        found *and* the audit had usable timestamps to inspect.
    """
    # Polymorphic entry: if the caller passed `train=`/`val=`/`test=`,
    # synthesise a flat sample list and split_assignments. The "timestamp"
    # field is also accepted as an alias for ``timestamp_field``.
    if samples is None and (train is not None or val is not None or test is not None):
        flat: list[Mapping[str, Any]] = []
        assignments: dict[int, str] = {}
        for split_name, split_samples in (("train", train), ("val", val), ("test", test)):
            if not split_samples:
                continue
            for s in split_samples:
                assignments[len(flat)] = split_name
                flat.append(s)
        samples = flat
        split_assignments = assignments
        # Detect short alias "timestamp" if the caller used that and the
        # default ``timestamp_field`` does not occur on the records.
        if flat and timestamp_field not in flat[0] and "timestamp" in flat[0]:
            timestamp_field = "timestamp"

    if samples is None or split_assignments is None:
        raise TypeError(
            "detect_temporal_leakage requires either (samples, split_assignments) "
            "or one of train=/val=/test= sample lists"
        )

    by_patient_split: dict[tuple[str, str], list[float]] = defaultdict(list)
    parse_failures = 0
    missing_field = 0

    for idx, sample in enumerate(samples):
        split = split_assignments.get(idx)
        if split is None:
            continue
        if patient_id_field not in sample or timestamp_field not in sample:
            missing_field += 1
            continue
        pid = str(sample[patient_id_field])
        ts = _coerce_timestamp(sample[timestamp_field])
        if ts is None:
            parse_failures += 1
            continue
        by_patient_split[(pid, split)].append(ts)

    leaks: list[dict[str, Any]] = []
    patients = {pid for (pid, _) in by_patient_split}
    pairs = (("train", "test"), ("train", "val"), ("val", "test"))

    for pid in patients:
        for early, late in pairs:
            early_times = by_patient_split.get((pid, early))
            late_times = by_patient_split.get((pid, late))
            if not early_times or not late_times:
                continue
            if max(early_times) >= min(late_times):
                leaks.append(
                    {
                        "patient_id": pid,
                        "train_time": max(early_times),
                        "test_time": min(late_times),
                        "split_pair": f"{early}->{late}",
                    }
                )

    details: dict[str, Any] = {
        "audited_n": len(samples),
        "timestamp_field": timestamp_field,
        "patient_id_field": patient_id_field,
        "parse_failures": parse_failures,
        "missing_field_count": missing_field,
        "patients_with_multi_split": sum(
            1
            for pid in patients
            if sum(
                1
                for s in ("train", "val", "test")
                if (pid, s) in by_patient_split
            )
            > 1
        ),
    }
    warnings: list[str] = []
    if not samples:
        warnings.append("no samples provided; temporal audit is vacuous")
    if missing_field:
        warnings.append(
            f"{missing_field} sample(s) missing {patient_id_field!r} or "
            f"{timestamp_field!r}; not audited"
        )
    if parse_failures:
        warnings.append(
            f"{parse_failures} timestamp(s) could not be parsed; not audited"
        )
    if warnings:
        details["warnings"] = warnings

    clean = not leaks and bool(samples) and not missing_field and not parse_failures
    if leaks:
        logger.error(
            "detect_temporal_leakage: %d patient/split pair(s) leak", len(leaks)
        )
    return LeakageReport(
        duplicate_patient_ids={},
        temporal_leaks=leaks,
        clean=clean,
        details=details,
    )


# ---------------------------------------------------------------------------
# Specimen alias leakage
# ---------------------------------------------------------------------------


def detect_specimen_alias_leakage(
    train_ids: Iterable[Any],
    test_ids: Iterable[Any],
    alias_map: Mapping[str, str],
) -> LeakageReport:
    """Detect same biological specimen under different IDs across splits.

    Common in DepMap datasets where a cell line appears as an ACH-* barcode
    in one split and its CCLE name in another. The ``alias_map`` maps each
    raw identifier to a canonical specimen ID. If the same canonical ID
    appears in both train and test, the specimen is flagged.

    Args:
        train_ids: Iterable of sample/specimen IDs in the training split.
        test_ids: Iterable of sample/specimen IDs in the test split.
        alias_map: Mapping ``raw_id -> canonical_id``. IDs not present in
            the map are assumed to already be canonical.

    Returns:
        A :class:`LeakageReport`. ``clean`` is ``True`` iff no canonical
        specimen appears in both splits.
    """
    train_list = _to_id_list(train_ids)
    test_list = _to_id_list(test_ids)

    def _resolve(raw: str) -> str:
        return str(alias_map.get(raw, raw))

    train_canonical: dict[str, list[str]] = defaultdict(list)
    for raw in train_list:
        train_canonical[_resolve(raw)].append(raw)

    test_canonical: dict[str, list[str]] = defaultdict(list)
    for raw in test_list:
        test_canonical[_resolve(raw)].append(raw)

    shared = set(train_canonical.keys()) & set(test_canonical.keys())

    duplicates: dict[str, list[str]] = {}
    leaks: list[dict[str, Any]] = []
    for canon in sorted(shared):
        train_raws = train_canonical[canon]
        test_raws = test_canonical[canon]
        duplicates[canon] = ["train", "test"]
        leaks.append(
            {
                "canonical_id": canon,
                "train_raw_ids": train_raws,
                "test_raw_ids": test_raws,
            }
        )

    details: dict[str, Any] = {
        "n_train": len(set(train_list)),
        "n_test": len(set(test_list)),
        "n_canonical_train": len(train_canonical),
        "n_canonical_test": len(test_canonical),
        "n_alias_collisions": len(shared),
    }
    warnings: list[str] = []
    if not train_list or not test_list:
        warnings.append("train or test split is empty; audit is vacuous")
    if not alias_map:
        warnings.append("alias_map is empty; only exact-match dedup applied")
    if warnings:
        details["warnings"] = warnings

    clean = not shared and bool(train_list) and bool(test_list)
    if shared:
        logger.error(
            "detect_specimen_alias_leakage: %d canonical specimen(s) overlap "
            "across splits",
            len(shared),
        )
    return LeakageReport(
        duplicate_patient_ids=duplicates,
        temporal_leaks=leaks,
        clean=clean,
        details=details,
    )


# ---------------------------------------------------------------------------
# Future covariate leakage
# ---------------------------------------------------------------------------


def detect_future_covariate_leakage(
    samples: Sequence[Mapping[str, Any]],
    features_used: Sequence[str],
    event_times: Mapping[str, float],
) -> LeakageReport:
    """Detect features measured after the prediction (event) time.

    For each sample, checks every feature in ``features_used`` against
    ``event_times``. If the feature's measurement time (stored as
    ``<feature>_time`` in the sample dict) is after the event time for that
    sample, it constitutes future covariate leakage.

    Args:
        samples: Sequence of sample dicts. Each must contain a
            ``"sample_id"`` key and optionally ``<feature>_time`` keys
            indicating when a feature was measured.
        features_used: List of feature names used as model inputs.
        event_times: Mapping ``sample_id -> event_time`` (float). The
            prediction horizon; features measured after this time leak.

    Returns:
        A :class:`LeakageReport`. ``clean`` is ``True`` iff no feature for
        any sample was measured after its event time.
    """
    leaks: list[dict[str, Any]] = []
    audited = 0

    for sample in samples:
        sid = str(sample.get("sample_id", ""))
        if not sid:
            continue
        evt = event_times.get(sid)
        if evt is None:
            continue
        audited += 1
        for feat in features_used:
            time_key = f"{feat}_time"
            feat_time = sample.get(time_key)
            if feat_time is None:
                continue
            ts = _coerce_timestamp(feat_time)
            if ts is None:
                continue
            if ts > evt:
                leaks.append(
                    {
                        "sample_id": sid,
                        "feature": feat,
                        "feature_time": ts,
                        "event_time": evt,
                        "delta": ts - evt,
                    }
                )

    details: dict[str, Any] = {
        "n_samples": len(samples),
        "n_features_checked": len(features_used),
        "n_audited": audited,
        "n_leaks": len(leaks),
    }
    warnings: list[str] = []
    if not samples:
        warnings.append("no samples provided; audit is vacuous")
    if not features_used:
        warnings.append("no features specified; nothing to check")
    if audited == 0 and samples:
        warnings.append("no samples had matching event_times; audit is vacuous")
    if warnings:
        details["warnings"] = warnings

    clean = not leaks and audited > 0
    if leaks:
        logger.error(
            "detect_future_covariate_leakage: %d feature/sample pair(s) "
            "measured after event time",
            len(leaks),
        )
    return LeakageReport(
        duplicate_patient_ids={},
        temporal_leaks=leaks,
        clean=clean,
        details=details,
    )


# ---------------------------------------------------------------------------
# Drug exposure leakage
# ---------------------------------------------------------------------------


def detect_drug_exposure_leakage(
    treatment_exposures: Sequence[Mapping[str, Any]],
    prediction_time: float,
    features_used: Sequence[str],
) -> LeakageReport:
    """Detect drug exposure after prediction time leaking into input features.

    Walks over ``treatment_exposures`` (each a dict with at least
    ``"drug_name"`` and ``"exposure_start"``). If a drug's exposure starts
    after ``prediction_time`` and the drug name (or a derivative feature like
    ``<drug>_dose``, ``<drug>_response``, ``<drug>_auc``) appears in
    ``features_used``, the pair is flagged.

    Args:
        treatment_exposures: Sequence of dicts, each with keys
            ``"drug_name"`` and ``"exposure_start"`` (numeric or parseable
            timestamp).
        prediction_time: The prediction horizon cutoff (float timestamp).
        features_used: Sequence of feature column names used as model input.

    Returns:
        A :class:`LeakageReport`. ``clean`` is ``True`` iff no post-cutoff
        drug exposure contaminates the feature set.
    """
    features_lower = {f.lower() for f in features_used}
    leaks: list[dict[str, Any]] = []
    audited = 0

    _SUFFIXES = ("", "_dose", "_response", "_auc", "_exposure", "_duration")

    for record in treatment_exposures:
        drug = str(record.get("drug_name", ""))
        if not drug:
            continue
        start_raw = record.get("exposure_start")
        start = _coerce_timestamp(start_raw)
        if start is None:
            continue
        audited += 1
        if start <= prediction_time:
            continue
        # Drug exposure is AFTER prediction time -- check features
        drug_lower = drug.lower()
        for suffix in _SUFFIXES:
            candidate = f"{drug_lower}{suffix}"
            if candidate in features_lower:
                leaks.append(
                    {
                        "drug_name": drug,
                        "exposure_start": start,
                        "prediction_time": prediction_time,
                        "leaking_feature": candidate,
                        "delta": start - prediction_time,
                    }
                )

    details: dict[str, Any] = {
        "n_exposures": len(treatment_exposures),
        "n_audited": audited,
        "prediction_time": prediction_time,
        "n_features": len(features_used),
        "n_leaks": len(leaks),
    }
    warnings: list[str] = []
    if not treatment_exposures:
        warnings.append("no treatment exposures provided; audit is vacuous")
    if not features_used:
        warnings.append("no features specified; nothing to check")
    if warnings:
        details["warnings"] = warnings

    clean = not leaks and audited > 0
    if leaks:
        logger.error(
            "detect_drug_exposure_leakage: %d post-cutoff drug/feature "
            "pair(s) found",
            len(leaks),
        )
    return LeakageReport(
        duplicate_patient_ids={},
        temporal_leaks=leaks,
        clean=clean,
        details=details,
    )


# ---------------------------------------------------------------------------
# Pseudotime / calendar time confusion
# ---------------------------------------------------------------------------


_PSEUDOTIME_SENTINELS = frozenset({
    "pseudotime_disease_stage",
    "pseudotime",
    "dpt_pseudotime",
    "diffusion_pseudotime",
})


def detect_pseudotime_calendar_confusion(
    pairs: Sequence[Mapping[str, Any]],
) -> LeakageReport:
    """Flag records where pseudotime values appear in calendar-time fields.

    Each entry in ``pairs`` is a dict with at least ``"time_field"`` (the
    name of the field being used as temporal ordering) and ``"time_value"``
    (its value). The check flags any record whose ``time_field`` matches a
    known pseudotime sentinel (e.g. ``"pseudotime_disease_stage"``) or whose
    value is a known pseudotime column name used as a literal value.

    This catches the subtle bug where ``pseudotime_disease_stage`` (an
    ordering metric in [0, 1]) is fed where calendar time (days since
    diagnosis) is expected, silently corrupting survival and temporal-split
    logic.

    Args:
        pairs: Sequence of dicts, each with ``"time_field"`` and
            ``"time_value"`` (and optionally ``"record_id"``).

    Returns:
        A :class:`LeakageReport`. ``clean`` is ``True`` iff no pseudotime
        value is found where calendar time is required.
    """
    leaks: list[dict[str, Any]] = []

    for idx, entry in enumerate(pairs):
        field_name = str(entry.get("time_field", "")).lower().strip()
        value = entry.get("time_value")
        record_id = entry.get("record_id", f"row_{idx}")

        flagged = False
        reason = ""

        # Check 1: the field name itself is a pseudotime sentinel
        if field_name in _PSEUDOTIME_SENTINELS:
            flagged = True
            reason = (
                f"time_field={field_name!r} is a pseudotime metric, not "
                f"calendar time"
            )

        # Check 2: the value is a string matching a pseudotime sentinel
        # (e.g. someone stored the column name as the value)
        if not flagged and isinstance(value, str):
            val_lower = value.lower().strip()
            if val_lower in _PSEUDOTIME_SENTINELS:
                flagged = True
                reason = (
                    f"time_value={value!r} is a pseudotime column name used "
                    f"as a literal value"
                )

        # Check 3: value in [0, 1] with a field name containing "pseudotime"
        if not flagged and "pseudotime" in field_name:
            ts = _coerce_timestamp(value) if not isinstance(value, str) else None
            if ts is not None and 0.0 <= ts <= 1.0:
                flagged = True
                reason = (
                    f"time_field={field_name!r} contains 'pseudotime' and "
                    f"value={value} is in [0,1], likely a pseudotime metric"
                )

        if flagged:
            leaks.append(
                {
                    "record_id": record_id,
                    "time_field": field_name,
                    "time_value": value,
                    "reason": reason,
                }
            )

    details: dict[str, Any] = {
        "n_pairs_checked": len(pairs),
        "n_confusions": len(leaks),
    }
    warnings: list[str] = []
    if not pairs:
        warnings.append("no time-field pairs provided; audit is vacuous")
    if warnings:
        details["warnings"] = warnings

    clean = not leaks and bool(pairs)
    if leaks:
        logger.error(
            "detect_pseudotime_calendar_confusion: %d record(s) use "
            "pseudotime where calendar time is required",
            len(leaks),
        )
    return LeakageReport(
        duplicate_patient_ids={},
        temporal_leaks=leaks,
        clean=clean,
        details=details,
    )


# ---------------------------------------------------------------------------
# External cohort overlap
# ---------------------------------------------------------------------------


def detect_external_cohort_overlap(
    train_ids: Iterable[Any],
    external_ids: Iterable[Any],
    cohort_name: str = "external",
) -> LeakageReport:
    """Check if external validation cohort overlaps with pretraining data.

    This catches the scenario where an "external" validation set is not
    truly external because some of its samples were part of the pretraining
    corpus (common when MMRF CoMMpass subsets appear in both DepMap and the
    user's training data).

    Args:
        train_ids: Iterable of patient/sample IDs in the training
            (or pretraining) set.
        external_ids: Iterable of patient/sample IDs in the external
            validation cohort.
        cohort_name: Human-readable label for the external cohort (used in
            diagnostics).

    Returns:
        A :class:`LeakageReport`. ``clean`` is ``True`` iff no ID appears
        in both sets.
    """
    train_list = _to_id_list(train_ids)
    ext_list = _to_id_list(external_ids)

    train_set = set(train_list)
    ext_set = set(ext_list)
    overlap = train_set & ext_set

    duplicates: dict[str, list[str]] = {
        pid: ["train", cohort_name] for pid in sorted(overlap)
    }

    details: dict[str, Any] = {
        "n_train": len(train_set),
        "n_external": len(ext_set),
        "cohort_name": cohort_name,
        "n_overlap": len(overlap),
    }
    warnings: list[str] = []
    if not train_list:
        warnings.append("train set is empty; audit is vacuous")
    if not ext_list:
        warnings.append(f"external cohort {cohort_name!r} is empty; audit is vacuous")
    if warnings:
        details["warnings"] = warnings

    clean = not overlap and bool(train_list) and bool(ext_list)
    if overlap:
        logger.error(
            "detect_external_cohort_overlap: %d ID(s) in %r also appear in "
            "training data",
            len(overlap),
            cohort_name,
        )
    return LeakageReport(
        duplicate_patient_ids=duplicates,
        temporal_leaks=[],
        clean=clean,
        details=details,
    )


# ---------------------------------------------------------------------------
# Full leakage audit
# ---------------------------------------------------------------------------


def run_full_leakage_audit(
    train_ids: Iterable[Any],
    val_ids: Iterable[Any],
    test_ids: Iterable[Any],
    samples: Sequence[Mapping[str, Any]] | None = None,
    alias_map: Mapping[str, str] | None = None,
    treatment_exposures: Sequence[Mapping[str, Any]] | None = None,
    external_cohorts: Mapping[str, Iterable[Any]] | None = None,
    *,
    features_used: Sequence[str] | None = None,
    event_times: Mapping[str, float] | None = None,
    prediction_time: float | None = None,
    time_field_pairs: Sequence[Mapping[str, Any]] | None = None,
    timestamp_field: str = "collection_date",
    patient_id_field: str = "patient_id",
) -> dict[str, Any]:
    """Run ALL leakage checks and return a comprehensive report.

    This is the single entry point for the leakage hard-stop gate. Every
    individual check is run; the overall verdict is PASS only if **all**
    checks pass.

    Args:
        train_ids: Training split IDs.
        val_ids: Validation split IDs.
        test_ids: Test split IDs.
        samples: Sample dicts for temporal and future-covariate checks.
        alias_map: For specimen alias check (raw_id -> canonical_id).
        treatment_exposures: For drug exposure check.
        external_cohorts: Mapping ``cohort_name -> iterable_of_ids`` for
            external overlap checks.
        features_used: Feature names for future-covariate and drug checks.
        event_times: For future-covariate check (sample_id -> event_time).
        prediction_time: For drug exposure check.
        time_field_pairs: For pseudotime/calendar confusion check.
        timestamp_field: Field name for temporal leakage check.
        patient_id_field: Field name for temporal leakage check.

    Returns:
        A dict with keys:
            ``"overall_pass"`` (bool),
            ``"checks"`` (dict mapping check name -> LeakageReport.to_dict()),
            ``"summary"`` (dict mapping check name -> "PASS" or "FAIL"),
            ``"n_passed"`` / ``"n_failed"`` / ``"n_skipped"`` (int).
    """
    checks: dict[str, LeakageReport] = {}

    # 1. Patient split overlap
    checks["patient_split"] = audit_patient_split(
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
    )

    # 2. Temporal leakage (requires samples)
    if samples is not None:
        split_assignments: dict[int, str] = {}
        _train_set = set(_to_id_list(train_ids))
        _val_set = set(_to_id_list(val_ids))
        _test_set = set(_to_id_list(test_ids))
        for idx, s in enumerate(samples):
            pid = str(s.get(patient_id_field, ""))
            if pid in _train_set:
                split_assignments[idx] = "train"
            elif pid in _val_set:
                split_assignments[idx] = "val"
            elif pid in _test_set:
                split_assignments[idx] = "test"
        checks["temporal_leakage"] = detect_temporal_leakage(
            samples=samples,
            split_assignments=split_assignments,
            timestamp_field=timestamp_field,
            patient_id_field=patient_id_field,
        )

    # 3. Specimen alias leakage
    if alias_map is not None:
        checks["specimen_alias"] = detect_specimen_alias_leakage(
            train_ids=train_ids, test_ids=test_ids, alias_map=alias_map,
        )

    # 4. Future covariate leakage
    if (
        samples is not None
        and features_used is not None
        and event_times is not None
    ):
        checks["future_covariate"] = detect_future_covariate_leakage(
            samples=samples,
            features_used=features_used,
            event_times=event_times,
        )

    # 5. Drug exposure leakage
    if (
        treatment_exposures is not None
        and prediction_time is not None
        and features_used is not None
    ):
        checks["drug_exposure"] = detect_drug_exposure_leakage(
            treatment_exposures=treatment_exposures,
            prediction_time=prediction_time,
            features_used=features_used,
        )

    # 6. Pseudotime / calendar confusion
    if time_field_pairs is not None:
        checks["pseudotime_calendar"] = detect_pseudotime_calendar_confusion(
            pairs=time_field_pairs,
        )

    # 7. External cohort overlaps (one check per cohort)
    if external_cohorts is not None:
        for cohort_name, ext_ids in external_cohorts.items():
            key = f"external_overlap_{cohort_name}"
            checks[key] = detect_external_cohort_overlap(
                train_ids=train_ids,
                external_ids=ext_ids,
                cohort_name=cohort_name,
            )

    # Build summary
    summary: dict[str, str] = {}
    n_passed = 0
    n_failed = 0
    for name, report in checks.items():
        if report.clean:
            summary[name] = "PASS"
            n_passed += 1
        else:
            summary[name] = "FAIL"
            n_failed += 1

    # Count skipped checks (those not triggered due to missing inputs)
    all_possible = {
        "patient_split", "temporal_leakage", "specimen_alias",
        "future_covariate", "drug_exposure", "pseudotime_calendar",
    }
    # external cohorts are dynamic, so we only count the named checks
    n_skipped = len(all_possible - set(checks.keys()))

    overall_pass = n_failed == 0 and n_passed > 0

    if not overall_pass:
        logger.error(
            "run_full_leakage_audit: FAIL (%d/%d checks passed, %d skipped)",
            n_passed,
            n_passed + n_failed,
            n_skipped,
        )
    else:
        logger.info(
            "run_full_leakage_audit: PASS (%d/%d checks passed, %d skipped)",
            n_passed,
            n_passed + n_failed,
            n_skipped,
        )

    return {
        "overall_pass": overall_pass,
        "checks": {name: r.to_dict() for name, r in checks.items()},
        "summary": summary,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# Split manifest freeze
# ---------------------------------------------------------------------------


def _sha256_ids(ids: Iterable[Any]) -> str:
    """Compute deterministic SHA-256 hash of a sorted list of string IDs."""
    id_list = sorted(_to_id_list(ids))
    payload = "\n".join(id_list).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_split_manifest(
    train_ids: Iterable[Any],
    val_ids: Iterable[Any],
    test_ids: Iterable[Any],
    seed: int | float | str,
    out_path: str | Path,
) -> Path:
    """Write a ``split_manifest.json`` with SHA-256 hashes of each split.

    The manifest captures the exact composition of each split so that any
    future run can verify it has not drifted. The file is human-readable
    JSON with hashes, counts, and the random seed.

    Args:
        train_ids: Training split IDs.
        val_ids: Validation split IDs.
        test_ids: Test split IDs.
        seed: The random seed used to produce the splits.
        out_path: Filesystem path to write the manifest JSON. Parent
            directory must exist.

    Returns:
        The resolved :class:`Path` to the written manifest file.
    """
    train_list = _to_id_list(train_ids)
    val_list = _to_id_list(val_ids)
    test_list = _to_id_list(test_ids)

    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "splits": {
            "train": {
                "n": len(train_list),
                "n_unique": len(set(train_list)),
                "sha256": _sha256_ids(train_list),
            },
            "val": {
                "n": len(val_list),
                "n_unique": len(set(val_list)),
                "sha256": _sha256_ids(val_list),
            },
            "test": {
                "n": len(test_list),
                "n_unique": len(set(test_list)),
                "sha256": _sha256_ids(test_list),
            },
        },
        "combined_sha256": _sha256_ids(
            list(train_list) + list(val_list) + list(test_list)
        ),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("freeze_split_manifest: wrote %s", out)
    return out
