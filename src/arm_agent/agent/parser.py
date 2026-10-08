"""Qwen3.5 XML tool-call parsing (E5-hardened).

The model emits:

    <tool_call>
    <function=move>
    <parameter=direction>+x</parameter>
    <parameter=distance_cm>3</parameter>
    </function>
    </tool_call>

Three rules baked in from E5 measurements (DESIGN.md 2.4):
  * regex, not JSON: values may be multi-line and unescaped;
  * `truncate_to_first_call` — even with eos_token_id fixed the 4bit stack can
    occasionally run past the turn, and a reply must never carry hallucinated
    follow-up turns back into the history;
  * the harness enforces "exactly one call per turn" and re-prompts on 0 or >1,
    because unordered multi-action execution is not what the tools were built for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([\w.\-]+)>")
_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    args: dict[str, str] = field(default_factory=dict)


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for block in _TOOL_CALL_RE.findall(text):
        match = _FUNCTION_RE.search(block)
        if not match:
            continue
        args = {m.group(1): m.group(2) for m in _PARAM_RE.finditer(block)}
        calls.append(ToolCall(name=match.group(1), args=args))
    return calls


def truncate_to_first_call(text: str) -> str:
    """Keep everything up to and including the first </tool_call> (if any)."""
    end = text.find("</tool_call>")
    if end == -1:
        return text
    return text[: end + len("</tool_call>")]


def as_float(value: str, default: float | None = None) -> float:
    """Tolerant numeric parse: the model sometimes writes '3cm' or '−30'."""
    if value is None:
        if default is None:
            raise ValueError("missing numeric argument")
        return default
    cleaned = re.sub(r"[^0-9eE+\-.]", "", str(value).replace("\u2212", "-"))
    try:
        return float(cleaned)
    except ValueError:
        if default is None:
            raise
        return default
