"""Decompose per-turn LLM latency: prefill vs decode vs vision vs empty_cache.

Run ONLY when the GPU is free (the LLM needs it; the sim server does not):

    cd arm_agent && bash scripts/agentenv.sh scripts/bench_llm_latency.py

WHY
---
M1 measured ~31 s/turn (before the adapter was fixed) and ~48 s/turn on the
first turns of the fixed run. Token accounting says a turn prefills only
~3000 tokens:

    tools 1427 + system 643 + history ~560 + 2 frames x 66 = ~2800

3000 tokens of prefill should not cost 30 s, so the bottleneck must be
MEASURED, not guessed. This script times each stage in isolation (the same
discipline that caught the move()/objects() bugs) so any optimisation targets
the term that actually dominates.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_DIR = "/lab/haoq_lab/cse12311731/qwen35_demo/models/Qwen3.5-9B"


def main() -> int:
    import argparse
    import torch
    from PIL import Image

    from arm_agent.agent.llm import QwenAgentModel
    from arm_agent.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=1, help="shard across N GPUs (device_map=auto)")
    ap.add_argument("--gpu-mem-gib", type=float, default=9.5, help="per-GPU budget for device_map")
    ap.add_argument("--fresh", action="store_true", help="short prompt only (skip the long-prompt cases)")
    args = ap.parse_args()

    m = QwenAgentModel(
        MODEL_DIR,
        load="4bit",
        gpu_mem_gib=args.gpu_mem_gib,
        max_new_tokens=160,
        n_gpu=args.gpus,
    )
    m.load_model()

    import torch as _t
    for i in range(_t.cuda.device_count()):
        used = _t.cuda.memory_allocated(i) / 2**30
        print(f"  GPU{i} allocated: {used:.2f} GiB")
    try:
        from collections import Counter

        dm = m._model.hf_device_map  # type: ignore[attr-defined]
        summary = dict(Counter(str(v) for v in dm.values()))
        print(f"  device_map summary: {summary}")
        offloaded = [k for k, v in dm.items() if "cpu" in str(v) or "disk" in str(v)]
        if offloaded:
            print(
                f"  [!] WARNING: {len(offloaded)} modules OFFLOADED to cpu/disk "
                f"(e.g. {offloaded[:3]}) -> per-turn GPU<->CPU transfers"
            )
        else:
            print("  [ok] all modules on GPU (no offload)")
    except Exception as exc:  # noqa: BLE001
        print(f"  (no hf_device_map: {exc})")

    img = Image.new("RGB", (256, 256), (120, 120, 120))
    task_text = (
        "任务：pick up the alphabet soup and place it in the basket\n"
        "[上图是全局相机看到的场景]\n机械爪位置 (-0.1522, -0.0066, 0.2487) m\n请开始。"
    )
    # ~8 filler history messages (the M1 cap is 10 messages total).
    hist = []
    for i in range(4):
        hist.append({"role": "assistant", "content": f"<tool_call><function=move><parameter=direction>+x</parameter></tool_call> #{i}"})
        hist.append({"role": "user", "content": f"[move +x 3.3cm] 实测位移 (0.0318, 0.0002, -0.0004) m（9 ticks）；机械爪现在位于 (-0.1203, -0.0064, 0.2483) #{i}"})

    short = [{"role": "system", "content": "你是助手。"}, {"role": "user", "content": "你好"}]
    full_noimg = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task_text}, *hist]
    full_img = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": task_text}]},
        *hist,
    ]

    def run(label: str, msgs, maxn: int) -> float:
        m.max_new_tokens = maxn
        t0 = time.perf_counter()
        out = m.chat(msgs, tools=TOOL_SCHEMAS)
        dt = time.perf_counter() - t0
        print(f"  {label:<46} {dt:7.2f}s   (out chars={len(out)})", flush=True)
        return dt

    print("=== LLM latency decomposition (RTX 2080 Ti, 4bit, no flash-attn) ===")
    t_short = run("A. short prompt, 1 new token  (pure model overhead)", short, 1)
    t_nohist_1 = run("B. full no-image, 1 new token  (prefill-only)", full_noimg, 1)
    t_nohist_40 = run("C. full no-image, 40 new tokens (prefill+decode)", full_noimg, 40)
    t_img_1 = run("D. full + 1 image, 1 new token", full_img, 1)
    t_img_40 = run("E. full + 1 image, 40 new tokens (real turn)", full_img, 40)

    # F: the M1 real budget -- TWO frames are kept (HISTORY_IMAGE_KEEP=2), and
    # the run's peak activation was measured at 8.17 GiB. If that pushes past
    # `max_memory`, transformers silently offloads layers to CPU and every turn
    # pays a GPU<->CPU transfer (the suspected source of the missing ~20 s/turn).
    img2 = Image.new("RGB", (256, 256), (130, 130, 130))
    full_img2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": task_text}]},
        *hist,
        {"role": "user", "content": [{"type": "image", "image": img2}, {"type": "text", "text": "[新画面] 机械爪已移动，请看最新图像。"}]},
    ]
    t_img2 = run("F. full + 2 images, 40 new tokens (M1 real budget)", full_img2, 40)

    print("\n=== derived ===")
    print(f"  prefill (B - A)                 : {t_nohist_1 - t_short:6.2f}s")
    print(f"  decode 39 tokens (C - B)        : {t_nohist_40 - t_nohist_1:6.2f}s "
          f"=> {(t_nohist_40 - t_nohist_1) / 39:.3f}s/token")
    print(f"  vision tower (D - B)            : {t_img_1 - t_nohist_1:6.2f}s")
    print(f"  REAL TURN (E)                   : {t_img_40:6.2f}s")
    print(f"  REAL TURN (F, 2 frames)         : {t_img2:6.2f}s   <-- M1 actual budget")
    print(f"  2nd frame cost (F - E)          : {t_img2 - t_img_40:6.2f}s")

    # empty_cache cost, isolated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        for _ in range(10):
            torch.cuda.empty_cache()
        print(f"  torch.cuda.empty_cache() x10    : {time.perf_counter() - t0:6.3f}s")
        print(f"  peak reserved                   : {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
