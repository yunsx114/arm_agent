"""Quantify the LLM's per-turn token budget — CPU only, no GPU needed.

    cd arm_agent && bash scripts/agentenv.sh scripts/bench_token_budget.py

WHY THIS MATTERS
----------------
On this GPU attention has NO memory-efficient SDPA kernel (measured: the
allocator asked for one full N x N matrix of 2900^2 x 16 heads x 2 B), so the
per-turn prefill cost grows with the SQUARE of the token count while decode is
cheap (a tool call is ~40 tokens). Optimising therefore means shrinking the
tokens fed to `generate()` each turn, not the generation length.

This script splits that budget into:
  * FIXED  : system prompt + 11 tool schemas (paid every single turn)
  * IMAGE  : what one attached 160px frame actually costs the processor
  * HISTORY: the growing transcript, measured against the M1 caps
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_DIR = "/lab/haoq_lab/cse12311731/qwen35_demo/models/Qwen3.5-9B"


def main() -> int:
    from transformers import AutoProcessor, AutoTokenizer

    from arm_agent.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    # ---------------------------------------------------------------- fixed
    sys_msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "任务：pick up the alphabet soup\n请开始。每次只调用一个工具。"},
    ]
    rendered_sys = tok.apply_chat_template(
        sys_msg, tools=TOOL_SCHEMAS, tokenize=False, enable_thinking=False, add_generation_prompt=True
    )
    n_sys = len(tok(rendered_sys).input_ids)
    n_sys_notools = len(
        tok(
            tok.apply_chat_template(
                sys_msg, tokenize=False, enable_thinking=False, add_generation_prompt=True
            )
        ).input_ids
    )
    print("=== FIXED overhead (paid EVERY turn) ===")
    print(f"  system prompt alone      : {n_sys_notools:>6} tokens")
    print(f"  system prompt + 11 tools : {n_sys:>6} tokens   (tools cost {n_sys - n_sys_notools})")
    print(f"  rendered chars           : {len(rendered_sys)}")

    # ---------------------------------------------------------------- image
    try:
        import torch
        from PIL import Image

        proc = AutoProcessor.from_pretrained(MODEL_DIR)
        plain = [{"role": "user", "content": "任务：pick up the alphabet soup\n请开始。"}]
        text_only = tok.apply_chat_template(plain, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        n_text = len(tok(text_only).input_ids)
        with_img = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "任务：pick up the alphabet soup\n请开始。"}]}
        ]
        text_img = tok.apply_chat_template(with_img, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        print(f"  (text-only reference: {n_text} tokens; rendered-with-image chars {len(text_img)} vs {len(text_only)})")
        for side in (160, 256):
            img = Image.new("RGB", (side, side), (128, 128, 128))
            try:
                inputs = proc(text=[text_img], images=[img], return_tensors="pt")
                n_img = int(inputs["input_ids"].shape[1])
                print("\n=== IMAGE cost (processor, includes placeholders) ===")
                print(f"  {side}x{side} frame: {n_img:>6} tokens  (text-only {n_text}, so +{n_img - n_text} per frame)")
            except Exception as exc:  # noqa: BLE001
                print(f"  {side}px failed: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[image cost skipped: {type(exc).__name__}: {exc}]")

    # -------------------------------------------------------------- history
    print("\n=== HISTORY (measured from the last real episode) ===")
    eps = sorted((REPO / "outputs" / "episodes").glob("episode_*.json"))
    if not eps:
        print("  no episode transcripts found")
        return 0
    d = json.loads(eps[-1].read_text())
    total = 0
    per_turn = []
    for item in d["transcript"]:
        txt = item.get("assistant") or item.get("result") or item.get("tool_error") or ""
        n = len(tok(str(txt)).input_ids)
        total += n
        per_turn.append((item.get("turn"), n))
    print(f"  transcript: {eps[-1].name}")
    print(f"  text tokens over {len(per_turn)} items: {total}")
    avg = total / max(1, len(per_turn))
    print(f"  average per item: {avg:.0f} tokens")
    # M1 caps: MAX_HISTORY_MESSAGES=10 messages (5 exchanges), HISTORY_IMAGE_KEEP=2
    print("\n=== per-turn prefill estimate (M1 caps: 10 msgs, 2 frames) ===")
    est = n_sys + 2 * 200 + int(avg * 10)
    print(f"  system+tools {n_sys} + ~2 frames + ~10 msgs x {avg:.0f} tokens")
    print(f"  => approx {est} tokens prefilled EVERY turn")
    print("  (note: cost ~ quadratic in this number on this card)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
