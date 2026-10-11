"""Strata backend: an OpenAI-compatible HTTP client for the harness.

`StrataAgentModel` is a drop-in replacement for `QwenAgentModel` behind the
harness's `model.chat(messages, tools, *, thinking, max_new_tokens)` contract, so
nothing in the loop, the parser or the recorder changes.  Strata (a local
MoE-serving engine, ~/autodl-tmp/Strata) runs Qwen3.8-Flash-Next and serves
`/v1/chat/completions` on 127.0.0.1:8080.

Three conversions, all measured against the live server (2026-10-11):

1. **Images.** The harness attaches PIL images as
   `{"type": "image", "image": <PIL>}`; Strata takes OpenAI `image_url` parts with
   base64 data URLs.  Verified: two frames arrive as ~2 x 1024 prompt tokens and
   the model reads the scene (it reported "罐子位于篮口正上方, xy 误差 1.2mm" off a
   frame).  `--vision yes` must be on at install time or /health says
   `"images": false`.
2. **Tools.** The v2 surface in prompts.py is already OpenAI-shaped
   (`{"type":"function","function":{...}}`), so it goes over the wire unchanged and
   the server answers with standard `tool_calls` whose `arguments` is a JSON
   string (verified: `finish_reason: tool_calls`).
3. **Back to XML.** The reply is converted back into the `<tool_call>` /
   `<function=...>` / `<parameter=...>` text the existing
   `parse_tool_calls`/`truncate_to_first_call` already handle, so the transcript,
   the gate and the analysis scripts keep seeing exactly the format they saw from
   the 9B.  Argument values are stringified: `validate_tool_call` is typed
   `dict[str, str]` and `as_float` tolerates strings.

`thinking` maps onto the server's `reasoning_effort` (none|low|medium|high,
default high), which is how the old "chain-of-thought on/off" A/B is expressed
now -- no second copy of the model, no context switch to fake a second role.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "strata"


def _data_url(image: Any) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Harness messages -> OpenAI messages (PIL images -> base64 data URLs)."""
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append({"role": message["role"], "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                parts.append({"type": "image_url", "image_url": {"url": _data_url(item["image"])}})
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif isinstance(item, str):
                parts.append({"type": "text", "text": item})
        out.append({"role": message["role"], "content": parts})
    return out


def _xml_value(value: Any) -> str:
    """A JSON argument -> the text a <parameter> body can carry."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        # 15.0 would parse fine, but the 9B wrote "15" and several tool
        # implementations echo the argument back; keep the tidy form.
        return f"{value:g}"
    text = str(value)
    # A value that closes the tag it sits in would truncate the call; the XML
    # path was never able to carry one either.
    return text.replace("</parameter>", " ").replace("<parameter=", " ")


def to_xml_tool_call(call: dict[str, Any]) -> str:
    """One OpenAI tool call -> the Qwen XML block the parser expects."""
    fn = call.get("function") or {}
    name = str(fn.get("name") or "").strip()
    raw = fn.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        # A truncated or malformed call: emit the name with no parameters, so
        # validate_tool_call reports `missing:<key>` and the harness re-prompts
        # instead of the harness silently running a garbage action.
        args = {}
    if not isinstance(args, dict):
        args = {}
    lines = ["", "<tool_call>", f"<function={name}>"]
    for key, value in args.items():
        lines.append(f"<parameter={key}>{_xml_value(value)}</parameter>")
    lines += ["</function>", "</tool_call>"]
    return "\n".join(lines)


class StrataAgentModel:
    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        max_new_tokens: int = 384,
        reasoning_effort: str = "none",
        thinking_effort: str = "high",
        timeout_s: float = 900.0,
        api_key: str | None = None,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # /health lives at the server ROOT, not under /v1: a naive
        # f"{base_url}/health" becomes /v1/health, which is a 404, and a 404 here
        # means load_model() polls for the full 600 s and then raises. Cost one
        # smoke test (2026-10-11) -- the paths are not interchangeable.
        _root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.health_url = f"{_root}/health"
        self.model = model
        # Same reason as QwenAgentModel: a str reaching the request body fails
        # deep inside ("'<=' not supported ..."); coerce at the boundary.
        self.max_new_tokens = int(max_new_tokens)
        self.reasoning_effort = str(reasoning_effort).strip().lower()
        # What `thinking=True` means for this backend. The verifier asks for it
        # per call (Verifier thinking=True), the executor leaves it alone.
        self.thinking_effort = str(thinking_effort).strip().lower()
        self.timeout_s = float(timeout_s)
        self.api_key = api_key
        self.retries = max(0, int(retries))
        self.calls = 0
        # The run banner and any harness code read `enable_thinking` off the
        # model object; here the thinking level is the per-request effort string
        # the server understands, so expose it in the shape the loop expects.
        self.enable_thinking = self.reasoning_effort != "none"

    # ------------------------------------------------------------------ load
    def load_model(self) -> None:
        """Wait for the server. No weights to load here -- the engine owns them."""
        deadline = time.time() + 600.0
        last = ""
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self.health_url, timeout=20) as r:
                    info = json.load(r)
                if info.get("loaded"):
                    print(
                        f"[llm] strata ready: {info.get('model')} "
                        f"ctx={info.get('max_context')} images={info.get('images')}",
                        flush=True,
                    )
                    if not info.get("images"):
                        print(
                            "[llm] WARNING: the server has images OFF; the harness's camera "
                            "frames will be ignored (reinstall with --vision yes)",
                            flush=True,
                        )
                    return
                last = "not loaded yet"
            except Exception as exc:  # noqa: BLE001 - the server may still be starting
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(3.0)
        raise RuntimeError(f"Strata server at {self.health_url} never became ready ({last})")

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        thinking: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        budget = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if thinking is None:
            effort = self.reasoning_effort
        else:
            effort = self.thinking_effort if thinking else "none"

        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "max_tokens": budget,
            "reasoning_effort": effort,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        t0 = time.time()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions", data=payload, headers=headers
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:400].decode("utf-8", "replace")
                raise RuntimeError(f"Strata HTTP {exc.code}: {detail}") from exc
            except Exception as exc:  # noqa: BLE001 - one retry for a transient socket error
                last_error = exc
                if attempt >= self.retries:
                    raise RuntimeError(f"Strata request failed: {type(exc).__name__}: {exc}") from exc
                time.sleep(2.0)
        else:  # pragma: no cover - the loop always breaks or raises
            raise RuntimeError(f"Strata request failed: {last_error}")

        self.calls += 1
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        calls = message.get("tool_calls") or []
        timings = data.get("timings") or {}
        usage = data.get("usage") or {}
        gen_tok = timings.get("predicted_n") or usage.get("completion_tokens") or 0
        gen_ms = timings.get("predicted_ms") or 0.0
        prompt_tok = timings.get("prompt_n") or usage.get("prompt_tokens") or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        print(
            f"[llm] strata: prompt {prompt_tok}tok (cached {cached}) "
            f"gen {gen_tok}tok @ {1000.0 * gen_tok / gen_ms:.1f}tok/s "
            f"effort={effort} wall={time.time() - t0:.1f}s"
            + (f" finish={choice.get('finish_reason')}" if choice.get("finish_reason") != "stop" else ""),
            flush=True,
        )

        if not calls:
            # No tool call: hand the text back as-is. The harness counts 0 calls
            # and re-prompts (its existing protocol-error path), which is what we
            # want -- an un-tool'd turn must never be dispatched.
            return content

        # Exactly one call per turn is the harness's rule; extra calls are dropped
        # here and the parser would have rejected them anyway.
        return content + "\n" + to_xml_tool_call(calls[0])
