"""Small IO helpers."""
from __future__ import annotations
import json
from pathlib import Path


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def read_json(path):
    return json.loads(Path(path).read_text())
