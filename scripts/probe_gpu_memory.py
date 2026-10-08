"""Pinpoint the M1 CUDA OOM: where does GPU memory grow across turns?

Measured context: the first real episode OOM'd at turn 15-17, always inside
generate() with "Tried to allocate 266 MiB". 266 MiB is suspiciously close to
fp32 logits for one position (248320 vocab x 280 bytes...), so this probe
replicates the harness's per-turn shape and records, EVERY turn:

    allocated / reserved / peak  (torch.cuda.memory_stats)

with a conversation that grows the way the real one does (each turn appends
assistant text + a user text+image reply, and `prepare_for_model` keeps only
the newest 2 images as pixels). Three variants are compared to separate causes:

  A) as-is (keep=2 images, max_new_tokens=160)
  B) without passing logits_to_keep explicitly (transformers default path)
  C) with fewer images (keep=0: text only) - isolates the vision tower

Run on gpu026 via srun (needs the model, not the sim):

    srun --partition=rtx2080ti --account=gpulab02 --qos=rtx2080ti --nodes=1 \
         --gres=gpu:1 --time=00:30:00 --job-name=mem_probe \
         env PYTHONPATH=$PWD/src $HOME/miniconda3/envs/qwen35/bin/python -u \
         scripts/probe_gpu_memory.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.agent.harness import compact_history  # noqa: E402
from arm_agent.agent.llm import QwenAgentModel, prepare_for_model  # noqa: E402

MODEL_DIR = Path("/lab/haoq_lab/cse12311731/qwen35_demo/models/Qwen3.5-9B")
IMAGE = REPO / "outputs" / "smoke" / "e1_libero_object_0_agentview.png"

TURNS = 25


def mem(step: str) -> None:
    free, total = torch.cuda.mem_get_info()
    print(
        f"  [{step:<24s}] alloc={torch.cuda.memory_allocated() / 2**30:5.2f} GiB "
        f"reserved={torch.cuda.memory_reserved() / 2**30:5.2f} GiB "
        f"free={free / 2**30:5.2f} GiB",
        flush=True,
    )


def run_variant(name: str, keep_images: int, image_max_side: int = 160) -> None:
    from PIL import Image

    print(f"\n===== variant {name} (keep_images={keep_images}, image_side={image_max_side}) =====", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = QwenAgentModel(
        model_dir=str(MODEL_DIR),
        load="4bit",
        max_new_tokens=160,
        image_max_side=image_max_side,
    )
    model.load_model()
    mem("after load")

    image = Image.open(IMAGE).convert("RGB")
    messages = [
        {"role": "system", "content": "你是机械臂智能体。" * 50},  # ~ realistic system length
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "任务：把字母汤放进篮子。\n" + "状态占位。" * 20},
            ],
        },
    ]

    for turn in range(1, TURNS + 1):
        t0 = time.time()
        try:
            reply = model.chat(
                prepare_for_model(compact_history(messages), keep_images), tools=None
            )
        except torch.OutOfMemoryError:
            mem(f"OOM at turn {turn}")
            print(f"  >>> variant {name} OOMed at turn {turn}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return
        dt = time.time() - t0
        messages.append({"role": "assistant", "content": reply[:500]})
        # mimic the harness: every user turn carries a fresh image + state text
        content = []
        if keep_images > 0:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": "工具结果：已执行。机械爪位置 (0.1234, -0.2345, 0.0987) m。" * 5})
        messages.append({"role": "user", "content": content})
        if turn % 5 == 0 or turn == 1:
            mem(f"turn {turn} ({dt:.1f}s)")
            print(
                f"      msgs={len(messages)} "
                f"img_in_ctx={sum(1 for m in messages if isinstance(m.get('content'), list))}",
                flush=True,
            )

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"  >>> variant {name}: survived {TURNS} turns, peak alloc {peak:.2f} GiB")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA on this node")
        return 2
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    print(f"GPU total {total / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB")

    # One variant per process: a model that held memory when it OOMed is not
    # reliably freed, and the next load then fails with a bogus 7 GiB request
    # (measured). `python probe_gpu_memory.py B` selects a variant.
    variant = sys.argv[1].upper() if len(sys.argv) > 1 else "B"
    if variant == "A":
        run_variant("A-keep2-256px", keep_images=2, image_max_side=256)
    elif variant == "B":
        run_variant("B-keep2-160px-compact", keep_images=2, image_max_side=160)
    elif variant == "C":
        run_variant("C-text-only", keep_images=0, image_max_side=160)
    else:
        print(f"unknown variant {variant!r}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
