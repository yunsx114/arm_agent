#!/usr/bin/env python3
"""Static guard: every dispatch string in the episode loop must be a real tool.

WHY THIS EXISTS
---------------
Three commits in a row (P0c, P0d, and the one before this) fixed a warning that
never fired, and each time the cause was a string that did not match:

    P0c : `_dispatch` did not return `kind` at all      -> `_grasped` never True
    P0d : the `kind` key was added, but the loop tested  `"descend"`
          while the adapter reports                      `"descend_to"`
          -> "你手里已经夹着它" fired 0 times in TWO runs (P0d, P0e)
          while it applied at t5 in both.

A missing key is easy to spot. A *value* that belongs to a different vocabulary
is not: the code reads fine, the test passes, and the model just never hears the
warning. So this probe compares the two vocabularies mechanically.

WHAT IT CHECKS
--------------
1. The episode loop dispatches on the tool NAME (`outcome["tool"]`), not on the
   adapter's `kind` (a different vocabulary).
2. Every literal compared against `outcome.get("tool")` / `_tool` appears as a
   `"name"` in `TOOL_SCHEMAS` -- i.e. it is a tool the model can actually call.
   A typo here is exactly the bug above.
3. Every `"name"` in `TOOL_SCHEMAS` is handled by `_dispatch`'s if-chain, so a
   new tool cannot be added and silently fall through to the generic branch.

No GPU, no sim, no network: pure source analysis, so it runs anywhere and fast.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "src" / "arm_agent" / "agent" / "harness.py"
PROMPTS = REPO / "src" / "arm_agent" / "agent" / "prompts.py"

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    print(f"[FAIL] {msg}")
    failures.append(msg)


def tool_schema_names(src: str) -> set[str]:
    """Tool names advertised to the model, from the TOOL_SCHEMAS literal."""
    block = src.split("TOOL_SCHEMAS", 1)
    if len(block) != 2:
        return set()
    body = block[1]
    # Stop at the end of the literal: the next top-level assignment.
    m = re.search(r"\n[A-Z_][A-Z_0-9]*\s*[:=]", body)
    if m:
        body = body[: m.start()]
    # Nested schemas: the tool's own name is the FIRST "name" of each entry,
    # which in this file is written as `"name": "<tool>",` right after `{`.
    return set(re.findall(r'\{\s*\n?\s*"type":\s*"function",\s*\n?\s*"function":\s*\{\s*\n?\s*"name":\s*"(\w+)"', body)) or set(
        re.findall(r'"name":\s*"(\w+)"', body)
    )


def main() -> int:
    hsrc = HARNESS.read_text()
    psrc = PROMPTS.read_text()

    # Scope the vocabulary check to `run_episode` ONLY. `format_outcome` also
    # compares against `kind`, and that one is legitimate: it consumes the
    # adapter's own vocabulary and its literals (move / rotate / set_gripper) are
    # exactly the adapter's kind strings. A check that flagged those too produced
    # false failures the first time this probe was run -- a probe that cries wolf
    # is worse than no probe.
    body = hsrc.split("def run_episode", 1)
    loop = body[1].split("\n    def ", 1)[0] if len(body) == 2 else ""
    if not loop:
        bad("could not locate run_episode in harness.py")

    # ---- 1. vocabulary -----------------------------------------------------
    kind_cmp = re.findall(r'\b_?kind\w*\s*==\s*"(\w+)"', loop)
    tools = tool_schema_names(psrc)
    if kind_cmp:
        bad(f"run_episode still compares an adapter `kind` vocabulary string: {kind_cmp}")
    else:
        ok("run_episode does not compare raw adapter `kind` strings")

    if 'outcome.setdefault("tool", call.name)' not in loop:
        bad("run_episode does not tag the outcome with the tool name")
    else:
        ok("run_episode tags each outcome with `tool` = call.name")

    if re.search(r'outcome\.get\("kind"\)\s*==\s*"set_gripper"', loop):
        bad("the grasp-state update still keys off `kind`")
    else:
        ok("the grasp-state update no longer keys off `kind`")

    # ---- 2. the literals actually used -------------------------------------
    if not tools:
        bad("could not parse TOOL_SCHEMAS tool names")
    used = set(re.findall(r'(?:outcome\.get\("tool"\)|_tool)\s*==\s*"(\w+)"', loop))
    if not used:
        bad("no tool-name comparisons found in run_episode")
    else:
        ok(f"run_episode dispatches on tool names: {sorted(used)}")
        unknown = used - tools
        if unknown:
            bad(f"loop compares tool names that are NOT in TOOL_SCHEMAS: {sorted(unknown)}")
        else:
            ok("every tool name the loop tests exists in TOOL_SCHEMAS")

    # ---- 3. every tool is handled -----------------------------------------
    # Both `if name == "x":` and `if name in ("x", "y"):` are used here, so
    # match both -- missing the tuple form is what produced the second false
    # failure of this probe (`set_gripper`).
    handled = set(re.findall(r'name == "(\w+)"', hsrc))
    for group in re.findall(r'name in \(([^)]*)\)', hsrc):
        handled |= set(re.findall(r'"(\w+)"', group))
    unhandled = tools - handled
    if unhandled:
        bad(f"tool(s) in TOOL_SCHEMAS have no `_dispatch` branch: {sorted(unhandled)}")
    else:
        ok(f"every TOOL_SCHEMAS entry has a `_dispatch` branch ({len(tools)} tools)")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
