"""Subtype-aware multiple myeloma time-to-event modeling."""
from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    # Single source of truth: the installed package metadata (pyproject version).
    __version__ = version("mm-tte-opisd")
except PackageNotFoundError:  # running from a source tree without installed metadata
    __version__ = "0.2.0"
