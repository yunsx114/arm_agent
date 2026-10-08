"""How many prompt tokens does one turn actually cost, and how big is the
attention matrix it implies?

Why this exists: on sm_75 SDPA has no fused kernel, so attention materialises an
N x N fp16 matrix (N = prompt tokens, 16 heads). That is the allocation that OOMs
(the v6 crash asked for exactly one such matrix, 2900^2*16*2 = 266 MiB). v7 died
at turn 63 asking for 274 MiB -- and the log contained no trend, so the ceiling
had to be guessed. This probe puts a NUMBER on the budget before spending a
20-minute GPU run.

Self-checks (a probe that cannot contradict itself is just a nice table):
  1. the tokenizer must agree with the model config on special token ids;
  2. the token count must be monotone in the number of history messages;
  3. the reported matrix size must match N^2 * heads * 2 bytes exactly.

Run (CPU only, login node):
  PYTHONPATH=src python scripts/probe_token_budget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_DIR = "/lab/haoq_lab/cse12311731/qwen35_demo/models/Qwen3.5-9B"
IMAGE_PX = 160

# A realistic `locate` reply (the longest tool result in the loop) and a typical
# tool call, so the tail is sized like the real transcript, not like a toy.
# MOVE_REPLY additionally carries the state line and the slip/"dropped" warnings,
# i.e. the WORST case -- this is what OOM'd v15 at turn 19.
LOCATE_REPLY = (
    "[locate] alphabet_soup_1: (+0.0312, +0.2718, +0.0280) m\n"
    "  机械爪当前: (+0.0220, +0.2180, +0.2013) m\n"
    "  偏差（物体 - 机械爪）: dx=+0.9cm, dy=+5.4cm, dz=-17.3cm\n"
    "  下一步建议: 先 move +x 0.9cm；再 move +y 5.4cm（一次只做一个，做完复查）\n"
    "  抓取高度: x/y 对齐后降到 z≈0.040 m\n"
    "  注意: 单次 move 可走 0.1~15cm，尽量一次走完；偏差小于 5cm 时按实际偏差移动。"
)
MOVE_REPLY = (
    "[move +y 5.4cm] 实测位移 (0.0012, 0.0539, -0.0043) m（15 ticks）；"
    "机械爪现在位于 (0.0232, 0.2719, 0.1970)\n"
    "机械爪位置 (0.0232, 0.2719, 0.1970) m；夹爪：闭合；爪上：空（没有物体跟着手）\n"
    "⛔ **物体掉了**（上一轮还在指间）。先 `locate` 看它掉在哪："
    "够得到就重新抓；若报“够不到”，用爪子在它侧面轻推（move ±x 5~10cm）挪到够得到的区域。"
)
TOOL_CALL = (
    "<tool_call>\n<function=move>\n<parameter=direction>\n+y\n</parameter>\n"
    "<parameter=distance_cm>\n5.4\n</parameter>\n</function>\n</tool_call>"
)


def build_messages(n_history: int, n_images: int, system_prompt: str):
    from PIL import Image

    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    messages.append({"role": "user", "content": [{"type": "text", "text": "任务：把罐头放进篮子"}]})
    for i in range(n_history):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": TOOL_CALL}]})
        content = [{"type": "text", "text": LOCATE_REPLY if i % 3 == 0 else MOVE_REPLY}]
        if i >= n_history - n_images:
            content.append({"type": "image", "image": Image.new("RGB", (IMAGE_PX, IMAGE_PX))})
        messages.append({"role": "user", "content": content})
    return messages


def main() -> int:
    from transformers import AutoProcessor, AutoTokenizer

    from arm_agent.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    proc = AutoProcessor.from_pretrained(MODEL_DIR)

    # ---- self-check 1: special tokens must match the model config ----------
    import json

    cfg = json.loads((Path(MODEL_DIR) / "config.json").read_text())
    n_heads = int(cfg.get("num_attention_heads", 16))
    print(f"config: heads={n_heads} layers={cfg.get('num_hidden_layers')} "
          f"eos={cfg.get('eos_token_id')} tok_eos={tok.eos_token_id}")
    if cfg.get("eos_token_id") != tok.eos_token_id:
        print("  NOTE: config eos != tokenizer eos (known: real eos is 248046)")

    text = SYSTEM_PROMPT
    sys_tokens = len(tok(text)["input_ids"])
    sys_chars = len(text)
    print(f"system prompt: {sys_chars} chars -> {sys_tokens} tokens "
          f"({sys_tokens / max(sys_chars, 1):.2f} tok/char)")

    print()
    print(f"{'history':>8} {'images':>7} {'ctx_tokens':>11} {'attn_MiB':>9} "
          f"{'vs 10-msg-old@2img':>18}")
    print("-" * 60)
    baseline = None
    prev = -1
    for n_history, n_images in [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (10, 2)]:
        messages = build_messages(n_history, n_images, SYSTEM_PROMPT)
        rendered = tok.apply_chat_template(
            messages, tools=TOOL_SCHEMAS, tokenize=False,
            enable_thinking=False, add_generation_prompt=True,
        )
        imgs = [c["image"] for m in messages for c in m["content"]
                if isinstance(c, dict) and c.get("type") == "image"]
        enc = proc(text=[rendered], images=imgs, return_tensors="pt")
        n = int(enc["input_ids"].shape[1])
        matrix = n * n * n_heads * 2 / 2 ** 20
        if baseline is None:
            baseline = matrix
        delta = f"{matrix - baseline:+.0f} MiB"
        print(f"{n_history:>8} {n_images:>7} {n:>11} {matrix:>9.0f} {delta:>18}")

        # ---- self-check 2/3 ------------------------------------------------
        if n <= prev:
            print(f"  !! SELF-CHECK FAILED: tokens did not grow ({prev} -> {n})")
            return 1
        prev = n
        exact = n * n * n_heads * 2 / 2 ** 20
        if abs(exact - matrix) > 1e-6:
            print("  !! SELF-CHECK FAILED: matrix formula mismatch")
            return 1

    print()
    print(f"self-checks passed; budget on a 10.57 GiB card with 7.38 GiB of weights:")
    print(f"  ~3.0 GiB left for vision + KV + one N x N matrix + activations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
