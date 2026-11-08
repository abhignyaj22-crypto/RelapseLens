"""Modality loaders. Returns the raw per-patient tables for the run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class RawModalities:
    clinical: pd.DataFrame
    cytogenetics: pd.DataFrame | None
    omics: pd.DataFrame | None
    program_activity: pd.DataFrame | None
    provenance: pd.DataFrame | None
    # Optional per-visit (long) clinical-biomarker table for the clinical-anchored
    # longitudinal path. Kept separate from `clinical` (one-row-per-patient) so the
    # static pipeline is untouched; None when no serial data is on disk.
    serial_clinical: pd.DataFrame | None = None


def _read(path: str | None) -> pd.DataFrame | None:
    if path and Path(path).exists():
        df = pd.read_csv(path)
        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str)
        return df
    return None


def load_modalities(clinical_path: str, cytogenetics_path: str | None = None,
                    omics_path: str | None = None,
                    program_activity_path: str | None = None,
                    serial_clinical_path: str | None = None) -> RawModalities:
    clinical = _read(clinical_path)
    if clinical is None:
        raise FileNotFoundError(f"clinical table not found: {clinical_path}")
    prov_path = Path(clinical_path).parent / "cytogenetics_provenance.csv"
    return RawModalities(
        clinical=clinical,
        cytogenetics=_read(cytogenetics_path),
        omics=_read(omics_path),
        program_activity=_read(program_activity_path),
        provenance=_read(str(prov_path)),
        serial_clinical=load_serial_clinical(serial_clinical_path) if serial_clinical_path else None,
    )


def load_serial_clinical(path: str | None) -> pd.DataFrame | None:
    """Load the per-visit (long) clinical-biomarker table WITHOUT de-duplicating
    patient rows (each patient contributes many visits). Returns None if absent.

    Unlike `_read`/`load_modalities`, this deliberately preserves repeated
    `patient_id` rows (each patient contributes many visits) rather than collapsing
    them as the static path does. Returns None when no per-visit table is on disk,
    which is the norm under the public-only data mandate.
    """
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)
    if "visit_t_days" in df.columns:
        df = df.sort_values(["patient_id", "visit_t_days"], kind="mergesort").reset_index(drop=True)
    return df
