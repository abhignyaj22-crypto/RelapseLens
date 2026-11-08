"""Merge invariant: the trusted core stays independent of the platform layer.

The reconciled framework enforces a strict ONE-WAY dependency —
`resistancemap` (platform/breadth) may import `mm_tte_survival` (trusted survival/eval
core), but NEVER the reverse. This keeps the core's 146 tests insulated and makes the
dependency graph acyclic. This test AST-scans every mm_tte_survival module and fails if
any of them imports resistancemap.
"""
from __future__ import annotations

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "mm_tte_survival"


def _imports_resistancemap(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "resistancemap" or a.name.startswith("resistancemap.")
                   for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "resistancemap" or mod.startswith("resistancemap."):
                return True
    return False


def test_core_never_imports_platform():
    offenders = []
    for py in CORE.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        if _imports_resistancemap(tree):
            offenders.append(str(py.relative_to(CORE.parent.parent)))
    assert not offenders, (
        "mm_tte_survival (trusted core) must NOT import resistancemap (platform layer) — "
        f"one-way dependency violated by: {offenders}")
