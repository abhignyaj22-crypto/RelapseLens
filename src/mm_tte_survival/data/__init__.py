"""Data layer: loaders, contracts, cohort assembly, splits, provenance.

Legacy names (prepare_dataset, DatasetBundle, load_tables, _hash_split) are
re-exported from .dataset so existing imports (`from .data import ...`) keep
working after the refactor.
"""
from .dataset import prepare_dataset, DatasetBundle, load_tables, _hash_split
from .loaders import load_modalities, RawModalities
from .cohort import build_matched_cohort
from .splits import hash_split, stratified_event_split
from .contracts import validate_all_inputs
from .provenance import load_provenance, confirmatory_allowed

__all__ = [
    "prepare_dataset", "DatasetBundle", "load_tables", "_hash_split",
    "load_modalities", "RawModalities", "build_matched_cohort",
    "hash_split", "stratified_event_split", "validate_all_inputs",
    "load_provenance", "confirmatory_allowed",
]
