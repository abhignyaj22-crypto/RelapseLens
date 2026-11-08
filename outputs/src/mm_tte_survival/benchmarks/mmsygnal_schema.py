"""Schema validator for mmSYGNAL 141-program activity input.

Guards against the two known misuse modes: feeding RNA PCs (PC1..PC128) or the
repo's 10 curated `program_activity.csv` signatures into the mmSYGNAL models —
neither is mmSYGNAL/MINER program activity and both would yield invalid scores.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def validate_mmsygnal_program_activity(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "patient_id" not in df.columns:
        raise ValueError("mmSYGNAL program activity must contain patient_id")

    # Accept either 0..140 or program_0..program_140
    raw_cols = {str(i) for i in range(141)}
    prefixed_cols = {f"program_{i}" for i in range(141)}
    cols = set(df.columns)

    if raw_cols.issubset(cols):
        ordered = ["patient_id"] + [str(i) for i in range(141)]
        return df[ordered]

    if prefixed_cols.issubset(cols):
        renamed = df.rename(columns={f"program_{i}": str(i) for i in range(141)})
        ordered = ["patient_id"] + [str(i) for i in range(141)]
        return renamed[ordered]

    missing_raw = sorted(raw_cols - cols, key=lambda x: int(x))
    missing_prefixed = sorted(prefixed_cols - cols, key=lambda x: int(x.split("_")[1]))
    raise ValueError(
        "Invalid mmSYGNAL input. Expected 141 program activity columns "
        "labeled 0..140 or program_0..program_140. "
        f"Missing raw labels example: {missing_raw[:10]}; "
        f"missing prefixed labels example: {missing_prefixed[:10]}"
    )


def is_valid_mmsygnal_program_activity(path: str | Path) -> bool:
    """Non-raising check used by the benchmark BLOCKED guard."""
    try:
        if not Path(path).exists():
            return False
        validate_mmsygnal_program_activity(path)
        return True
    except Exception:
        return False
