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
PROTOCOL = REPO / "src" / "arm_agent" / "agent" / "tool_protocol.py"

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    print(f"[FAIL] {msg}")
    failures.append(msg)


def tool_schema_names(src: str, anchor: str = "TOOL_SCHEMAS") -> set[str]:
    """Tool names advertised to the model, from one TOOL_SCHEMAS literal.

    There are two surfaces now (v1 = `TOOL_SCHEMAS`, v2 = `TOOL_SCHEMAS_V2`), so
    the caller unions both: the loop compares CANONICAL names (`gripper`,
    `approach`) under v2 and legacy names (`set_gripper`, `descend_to`) under v1,
    and both must be recognised or this probe cries wolf on a correct file.
    """
    block = src.split(anchor, 1)
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
    hsrc_protocol = PROTOCOL.read_text()

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
    tools = tool_schema_names(psrc) | tool_schema_names(psrc, "TOOL_SCHEMAS_V2")
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

    legacy_tools = tool_schema_names(psrc)                       # v1 surface
    v2_tools = tool_schema_names(psrc, "TOOL_SCHEMAS_V2")         # v2 surface

    unhandled = legacy_tools - handled
    if unhandled:
        bad(f"tool(s) in TOOL_SCHEMAS have no `_dispatch` branch: {sorted(unhandled)}")
    else:
        ok(f"every TOOL_SCHEMAS entry has a `_dispatch` branch ({len(legacy_tools)} tools)")

    # v2 names are translated to the legacy names at the top of `_dispatch`, so a
    # v2 tool needs either its own branch OR an alias target that has one. Without
    # this the probe reports a false failure for observe/approach/gripper/verify --
    # i.e. exactly the four tools whose wiring it is supposed to protect.
    aliases_block = hsrc_protocol.split("ALIASES", 1)[1].split("}", 1)[0]
    aliases = dict(re.findall(r'"(\w+)":\s*"(\w+)"', aliases_block))
    unresolved = []
    for tool in sorted(v2_tools):
        targets = {tool} | {legacy for legacy, canon in aliases.items() if canon == tool}
        if not (targets & handled):
            unresolved.append(tool)
    if unresolved:
        bad(f"v2 tool(s) resolve to no `_dispatch` branch: {unresolved}")
    else:
        ok(f"every v2 tool resolves to a `_dispatch` branch ({len(v2_tools)} tools)")

    check_context_summary()

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


def check_context_summary() -> None:
    """The 32-turn self-written context compression (M2): arithmetic + plumbing.

    Runs the real `_summarise_context` against a STUB model -- no GPU, no sim, no network --
    because the mechanism has three failure modes that are all invisible in a run's summary
    line: it fires on the wrong turn, it hands the model a view it has not seen, or it
    silently replaces an episode's history with an empty string.
    """
    print("\n-- 上下文压缩（每 N 轮由模型自述）--")
    sys.path.insert(0, str(REPO / "src"))
    from arm_agent.agent.harness import (  # noqa: PLC0415
        CONTEXT_SUMMARY_PREFIX,
        SUMMARY_EVERY_TURNS,
        SUMMARY_REQUEST,
        Harness,
        summary_due,
    )

    fires = [t for t in range(1, 129) if summary_due(t, SUMMARY_EVERY_TURNS)]
    if fires == [33, 65, 97]:
        ok(f"触发点恰好是每块结束后的第一轮：{fires}（默认每 {SUMMARY_EVERY_TURNS} 轮）")
    else:
        bad(f"触发点错了：{fires}（应为 [33, 65, 97]）")
    if not any(summary_due(t, 0) for t in range(1, 130)):
        ok("SUMMARY_EVERY=0 时永不触发（A/B 对照臂）")
    else:
        bad("SUMMARY_EVERY=0 仍然触发了")

    needed = ("目前为止都做了什么", "卡住", "接下来要做什么", "不要调用任何工具")
    missing = [k for k in needed if k not in SUMMARY_REQUEST]
    if missing:
        bad(f"压缩请求文本缺少关键问题/约束：{missing}")
    else:
        ok("压缩请求包含三个问题 + 不许调用工具")

    class StubModel:
        def __init__(self, reply):
            self.reply = reply
            self.calls: list[dict] = []

        def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
            self.calls.append({"tools": tools, "n_messages": len(messages), "kw": kwargs})
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

    class StubClient:
        ipc_dir = REPO / "runtime"

    def build(model, **kw):
        return Harness(
            client=StubClient(), model=model, workspace=REPO, tool_protocol="v2",
            log=False, **kw,
        )

    # (a) a good reply -> exactly three messages, system + task + brief.
    good = "1. 已向 -y 走过 3 次共 3cm，夹爪试了 1 次夹空。\n2. 卡住：在 -y/+y 之间来回。\n3. 下一步目标在 -x，用 gap_px 报像素。"
    model = StubModel(good)
    harness = build(model, summary_every=32)
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": [{"type": "text", "text": "TASK"}]},
        *[{"role": "user", "content": [{"type": "text", "text": f"turn {i}"}]} for i in range(1, 33)],
    ]
    compressed, summary = harness._summarise_context(messages, turn=33, recorder=None)
    problems = []
    if compressed is None:
        problems.append("返回 None")
    else:
        if [m["role"] for m in compressed] != ["system", "user", "user"]:
            problems.append(f"角色序列 {[m['role'] for m in compressed]}")
        if compressed[0]["content"] != "SYS" or "TASK" not in str(compressed[1]["content"]):
            problems.append("system/task 没保住")
        note = str(compressed[2]["content"])
        if CONTEXT_SUMMARY_PREFIX not in note or "夹空" not in note:
            problems.append("摘要没进历史")
        if "截至第 32 轮" not in note:
            problems.append("没写清压缩到第几轮")
    if problems:
        bad(f"压缩结果不对：{problems}")
    else:
        ok("压缩后正好是 [system, 任务, 自述摘要]，且注明截至第 32 轮")
    if model.calls and model.calls[0]["tools"] is None:
        ok("摘要请求按 tools=None 发出（引擎不会回工具调用）")
    else:
        bad(f"摘要请求带了 tools：{model.calls[:1]}")
    if len(messages) == 34:
        ok("压缩不修改传入的 messages（原历史仍可被上层保留）")
    else:
        bad("压缩把传入的 messages 改掉了")

    # (b) failure paths must never destroy the conversation.
    for label, reply in (("模型报错", RuntimeError("boom")), ("空回复", "   ")):
        h = build(StubModel(reply), summary_every=32)
        out, text = h._summarise_context(messages, turn=33, recorder=None)
        if out is None and text == "":
            ok(f"{label} → 保留原历史（返回 None）")
        else:
            bad(f"{label} 仍替换了历史：{str(out)[:60]}")

    # (c) a model that ignores "no tools" must not leak XML into the brief.
    toolish = (
        "1. 已经向 -y 走过 3 次共 3cm，夹爪试了 1 次夹空。\n"
        "2. 没有卡住，只是每步太小。\n"
        "3. 下一步继续 -y。\n"
        "<tool_call>\n<function=move>\n</function>\n</tool_call>"
    )
    h = build(StubModel(toolish), summary_every=32)
    out, text = h._summarise_context(messages, turn=33, recorder=None)
    if out is not None and "<" not in text and "夹空" in text:
        ok("回复里混进的工具调用被剥掉，摘要里不留 XML")
    else:
        bad(f"工具调用残留进了摘要：{text[:80]!r}")

    # (d) the message window must hold a whole block, or the brief is silently partial.
    auto = build(StubModel(good), summary_every=32)
    if auto.history_messages >= 2 * 32 + 8:
        ok(f"压缩开启且未指定窗口时自动放大到 {auto.history_messages} 条（≥ 一整块）")
    else:
        bad(f"窗口太小会截断摘要输入：{auto.history_messages}")
    off = build(StubModel(good), summary_every=0)
    if off.summary_every == 0 and off.history_messages == 32:
        ok("关闭压缩时窗口保持原来的 32 条")
    else:
        bad(f"关闭压缩却改了窗口：every={off.summary_every} history={off.history_messages}")


if __name__ == "__main__":
    raise SystemExit(main())
