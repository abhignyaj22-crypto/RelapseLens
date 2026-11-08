"""Time-axis (clock) registry loader.

Every endpoint resolves to exactly one clock (configs/time_axes.yaml). The clock records
whether it is serial (repeated per-patient observations), whether the underlying data is
controlled-access, and whether it may license a per-patient calendar-time forecast. This
prevents a run from silently conflating an event clock (OS/PFS/TT2L), the serial clinical
*observation* clock, and molecular pseudotime — see docs/roadmap.md.
"""
from __future__ import annotations

from pathlib import Path
import yaml

# endpoint_type (endpoints.yaml) -> time-axis name (time_axes.yaml)
_ENDPOINT_TYPE_TO_AXIS = {
    "overall_survival": "overall_survival",
    "progression_free_survival": "progression",
    "ttnt_proxy": "time_to_next_treatment",
    "early_progression_landmark": "progression",
    "clinically_anchored_longitudinal": "time_to_next_treatment",
}


def load_time_axes(path: str | Path = "configs/time_axes.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


def resolve_time_axis(endpoint_spec: dict, path: str | Path = "configs/time_axes.yaml") -> dict:
    """Return the clock spec for a run's endpoint. `endpoint_spec` may name the axis
    explicitly (`time_axis:`), else it is inferred from `endpoint_type`."""
    reg = load_time_axes(path)
    name = endpoint_spec.get("time_axis") or _ENDPOINT_TYPE_TO_AXIS.get(
        endpoint_spec.get("endpoint_type", ""), "overall_survival")
    spec = dict(reg.get(name, {}))
    spec["name"] = name
    return spec
