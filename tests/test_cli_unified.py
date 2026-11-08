"""A1 1.6 — the unified what/when/why/serve CLI, driven by the goal registry.

Exercises the list/status paths and the HONEST refusal: a BLOCKED capability must not run (exit 3), an
unknown capability errors (exit 2). Dispatch (`--run` of a live capability) and `serve` launch real
subprocesses/servers, so they are not exercised here — the point is that the CLI enumerates the honest
scoreboard and refuses to fabricate blocked results.
"""
from __future__ import annotations

from resistancemap.cli import main
from resistancemap.goal import Question, Status, capabilities_for


def test_status_runs():
    assert main(["status"]) == 0


def test_each_question_lists():
    for q in ("what", "when", "why"):
        assert main([q]) == 0            # default = list
        assert main([q, "--list"]) == 0


def test_blocked_capability_refuses_to_run():
    blocked = [c for c in capabilities_for(Question.WHEN) if c.status == Status.BLOCKED]
    assert blocked, "expected at least one BLOCKED WHEN capability (dynamic trajectory)"
    assert main(["when", "--run", blocked[0].name]) == 3     # honest refusal, not a fabricated result


def test_unknown_capability_errors():
    assert main(["when", "--run", "does-not-exist"]) == 2
