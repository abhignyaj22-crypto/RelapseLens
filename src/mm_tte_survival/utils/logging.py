"""Lightweight stdout logger."""
from __future__ import annotations
import sys


def log(msg: str) -> None:
    print(f"[mm-tte] {msg}", file=sys.stderr, flush=True)
