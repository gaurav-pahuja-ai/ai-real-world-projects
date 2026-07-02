"""
Eval Harness for the Research Agent
====================================
A miniature CI-for-agents. Two layers, run in this order:

1. Unit checks on parse_action() — deterministic, free, no API calls.
   Catches regressions in the ReAct output parser itself, which is the
   most common silent-failure point in text-parsed agents.

2. Task-level checks that run the full agent end to end and grade the
   report against expected tool calls and expected content. Costs real
   API calls, so it is opt-in via --live.

Usage:
    python eval.py            # parser unit checks only (free, ~instant)
    python eval.py --live     # also runs the full agent on real tasks (costs $)

This is the same shape of harness described in the "Evaluating and
Testing Agents" lesson: pick realistic tasks, define what a pass looks
like, run it on every change, treat it like CI rather than one-off QA.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field

from agent import parse_action, run_agent


# ── Layer 1: parser unit checks (no API calls) ───────────────────────────────

PARSER_CASES = [
    {
        "name": "simple search action",
        "input": 'Thought: I need current data.\nAction: search\nAction Input: {"query": "AAPL price"}',
        "expected_action": "search",
        "expected_input": {"query": "AAPL price"},
    },
    {
        "name": "finish action with report",
        "input": 'Thought: Done.\nAction: finish\nAction Input: {"report": "# Summary\\nDone."}',
        "expected_action": "finish",
        "expected_input": {"report": "# Summary\nDone."},
    },
    {
        "name": "malformed JSON input degrades to empty dict, not a crash",
        "input": "Thought: trying.\nAction: search\nAction Input: {not valid json}",
        "expected_action": "search",
        "expected_input": {},
    },
    {
        "name": "no action present returns None",
        "input": "Thought: still thinking, no action yet.",
        "expected_action": None,
        "expected_input": None,
    },
]


def run_parser_checks() -> bool:
    print("=== Layer 1: parser unit checks (free, no API calls) ===\n")
    passed = 0
    for case in PARSER_CASES:
        result = parse_action(case["input"])
        if case["expected_action"] is None:
            ok = result is None
        else:
            ok = (
                result is not None
                and result[0] == case["expected_action"]
                and result[1] == case["expected_input"]
            )
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {case['name']}")
        if not ok:
            print(f"         got: {result}")

    total = len(PARSER_CASES)
    print(f"\n  {passed}/{total} parser checks passed\n")
    return passed == total


# ── Layer 2: end-to-end task checks (real API calls, opt-in) ────────────────

@dataclass
class TaskCase:
    task: str
    expected_keywords: list[str] = field(default_factory=list)
    max_seconds: float = 90.0


TASK_CASES = [
    TaskCase(
        task="What is RAG in AI and how does it work?",
        expected_keywords=["retrieval", "generation"],
    ),
    TaskCase(
        task="What is the Model Context Protocol (MCP) and who created it?",
        expected_keywords=["mcp", "anthropic"],
    ),
]


def run_task_checks() -> bool:
    print("=== Layer 2: end-to-end task checks (real API calls) ===\n")
    passed = 0
    for case in TASK_CASES:
        start = time.monotonic()
        report = run_agent(case.task)
        elapsed = time.monotonic() - start

        report_lower = report.lower()
        missing = [kw for kw in case.expected_keywords if kw.lower() not in report_lower]
        within_time = elapsed <= case.max_seconds

        ok = not missing and within_time
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        print(f"  [{status}] {case.task!r}  ({elapsed:.1f}s)")
        if missing:
            print(f"         missing expected keywords: {missing}")
        if not within_time:
            print(f"         exceeded time budget: {elapsed:.1f}s > {case.max_seconds}s")

    total = len(TASK_CASES)
    print(f"\n  {passed}/{total} task checks passed\n")
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval harness for the research agent")
    parser.add_argument("--live", action="store_true", help="Also run end-to-end task checks (costs API calls)")
    args = parser.parse_args()

    parser_ok = run_parser_checks()

    if args.live:
        tasks_ok = run_task_checks()
    else:
        print("Skipping live task checks (pass --live to run them against real APIs)\n")
        tasks_ok = True

    sys.exit(0 if (parser_ok and tasks_ok) else 1)
