"""Unified `what | when | why | serve` CLI for the dashboard goal (A1 1.6).

ONE coherent entrypoint over the existing runners, driven by the goal registry (`resistancemap.goal`) —
so you no longer have to remember which of `make resistancemap` / `make mortfm` / `scripts/autoresearch/
run_*.py` to invoke. Each of the three questions lists its capabilities with honest status/tier/evidence
and dispatches `--run <name>` to that capability's autoresearch keep-revert entry. BLOCKED capabilities
refuse to run and print why (never a fabricated result).

    resistancemap status                 # the honest scoreboard (what's real vs blocked)
    resistancemap when --list            # capabilities for the WHEN question
    resistancemap what --run per-drug…    # dispatch a capability's runner
    resistancemap serve [--host --port]  # launch the FastAPI /predict surface

This is additive: the legacy runners still work; the Makefile gains `what/when/why/serve` targets.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from resistancemap.goal import CAPABILITIES, Question, Status, capabilities_for, summary

ROOT = Path(__file__).resolve().parents[2]
_Q = {"what": Question.WHAT, "when": Question.WHEN, "why": Question.WHY}


def _print_capabilities(question: Question) -> None:
    for c in capabilities_for(question):
        mark = {"answerable_now": "✓", "partial": "~", "blocked_pending_data": "✗"}[c.status.value]
        run = c.autoresearch_entry or "(no runner)"
        print(f"  {mark} [{c.dashboard_layer}] {c.name}\n"
              f"      status={c.status.value} tier={c.tier.value}\n"
              f"      evidence: {c.evidence}\n"
              f"      run: {run}" + (f"\n      BLOCKED: {c.blocked_reason}" if c.blocked_reason else ""))


def _cmd_status(_args) -> int:
    print("ResistanceMap — honest capability scoreboard (goal.py):\n")
    for q, counts in summary().items():
        print(f"  {q}: " + ", ".join(f"{k.split('_')[0]}={v}" for k, v in counts.items()))
    print()
    for q in Question:
        print(f"[{q.value}]")
        _print_capabilities(q)
    return 0


def _cmd_question(question: Question, args) -> int:
    if not getattr(args, "run", None):
        print(f"[{question.value}] capabilities:")
        _print_capabilities(question)
        return 0
    match = [c for c in capabilities_for(question) if c.name == args.run]
    if not match:
        names = [c.name for c in capabilities_for(question)]
        print(f"unknown capability '{args.run}'. choices: {names}", file=sys.stderr)
        return 2
    cap = match[0]
    if cap.status == Status.BLOCKED:
        print(f"REFUSING to run BLOCKED capability '{cap.name}': {cap.blocked_reason}\n"
              f"(needs data tier {cap.tier.value}; not fabricating a result.)", file=sys.stderr)
        return 3
    if not cap.autoresearch_entry:
        print(f"'{cap.name}' has no runner wired yet.", file=sys.stderr)
        return 4
    cmd = [sys.executable, str(ROOT / cap.autoresearch_entry), *args.extra]
    print("dispatch:", " ".join(cmd))
    return subprocess.run(cmd).returncode


def _cmd_serve(args) -> int:
    # serve the HONEST surface (A1 3.1): claim-tier per response, blocked claims refused — NOT the
    # aspirational api.py that needs absent weights and would serve blocked/refuted claims.
    from resistancemap.inference.honest_api import create_honest_app
    app = create_honest_app()
    if args.dry_run:
        print(f"would serve the honest what/when/why surface on {args.host}:{args.port}")
        return 0
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="resistancemap",
                                 description="what | when | why | serve — the dashboard goal, honestly gated")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="honest scoreboard of every capability").set_defaults(func=_cmd_status)
    for name in ("what", "when", "why"):
        p = sub.add_parser(name, help=f"the {name!r} question")
        p.add_argument("--list", action="store_true", help="list capabilities (default)")
        p.add_argument("--run", metavar="NAME", help="dispatch a capability's runner")
        p.add_argument("extra", nargs=argparse.REMAINDER, help="args forwarded to the runner")
        p.set_defaults(func=lambda a, _n=name: _cmd_question(_Q[_n], a))
    ps = sub.add_parser("serve", help="launch the FastAPI /predict surface")
    ps.add_argument("--host", default="127.0.0.1"); ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--dry-run", action="store_true", help="print target instead of launching")
    ps.set_defaults(func=_cmd_serve)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
