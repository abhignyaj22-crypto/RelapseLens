"""Modality abstraction: (name, domain, schema, loader) — Stage B/C."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import pandas as pd

Domain = Literal["patient", "cell_line", "network"]


class Modality(ABC):
    """A data modality with domain scope and feature extraction."""

    name: str
    domain: Domain

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load raw modality table."""

    @abstractmethod
    def to_features(self) -> pd.DataFrame:
        """Return model-ready feature matrix (patient_id or sample_id indexed)."""


class TranscriptomicsModality(Modality):
    name = "transcriptomics"
    domain: Domain = "patient"

    def __init__(self, path: Path, id_col: str = "patient_id"):
        self.path = Path(path)
        self.id_col = id_col

    def load(self) -> pd.DataFrame:
        if self.path.suffix == ".npz":
            import numpy as np
            z = np.load(self.path, allow_pickle=True)
            return pd.DataFrame(z["matrix"], index=z["patient_ids"], columns=z["genes"])
        return pd.read_csv(self.path)

    def to_features(self) -> pd.DataFrame:
        df = self.load()
        if self.id_col in df.columns:
            return df.set_index(self.id_col)
        return df


class ClinicalModality(Modality):
    name = "clinical"
    domain: Domain = "patient"

    def __init__(self, path: Path, id_col: str = "patient_id"):
        self.path = Path(path)
        self.id_col = id_col

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def to_features(self) -> pd.DataFrame:
        df = self.load()
        return df.set_index(self.id_col)


class ProgramActivityModality(Modality):
    name = "program_activity"
    domain: Domain = "patient"

    def __init__(self, path: Path, id_col: str = "patient_id"):
        self.path = Path(path)
        self.id_col = id_col

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def to_features(self) -> pd.DataFrame:
        return self.load().set_index(self.id_col)


class ProteomicsModality(Modality):
    """Cell-line surrogate — never claimed as patient measurement."""

    name = "proteomics"
    domain: Domain = "cell_line"

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Proteomics surrogate not fetched: {self.path}")
        return pd.read_csv(self.path)

    def to_features(self) -> pd.DataFrame:
        return self.load()


class EpigenomicsModality(Modality):
    name = "epigenomics"
    domain: Domain = "cell_line"

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Epigenomics surrogate not fetched: {self.path}")
        return pd.read_csv(self.path)

    def to_features(self) -> pd.DataFrame:
        return self.load()


class PPINetworkModality(Modality):
    name = "ppi"
    domain: Domain = "network"

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path, sep=" ")

    def to_features(self) -> pd.DataFrame:
        return self.load()
