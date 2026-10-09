#!/usr/bin/env python3
"""Which architectural levers are real, and which are folklore?

Written to answer "what else can be optimised (and what would a 32GB Blackwell
card change)". Each section produces a NUMBER, because the previous round showed
how expensive an un-measured belief is: `attn_matrix≈` (a formula, never
measured) under-reported the real transient by 5.3x for weeks.

THREE QUESTIONS
  1. Does this card actually have a fused attention backend? If flash /
     memory-efficient SDPA are unavailable, the fp32 math fallback materialises
     the score matrix -- which is the measured 170.7 B/tok^2, not the 32 B/tok^2
     the analytic formula assumes. `torch.backends.cuda.can_use_*_attention`
     answers this directly instead of by inference from memory numbers.
  2. Where do the 7.34 GiB of weights actually go? A 9B parameter model at 4bit
     should be ~4.5 GiB; if `lm_head` is left in fp16 (bitsandbytes does not
     quantise it by default, and this checkpoint has tie_word_embeddings=False so
     lm_head is a real 1.02B-parameter matrix) that alone is ~2 GiB.
  3. How much does PREFIX REUSE save? The system prompt + 11 tool schemas are
     re-sent every single turn and re-prefilled from scratch. Measuring
     fresh-prefill vs incremental-prefill on the same tokens says whether a KV
     cache is worth wiring up -- and unlike everything else here it would help on
     the CURRENT card.

Run (GPU node): bash scripts/run_arch_levers_on_gpu.sh
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_DIR = os.environ.get(
    "ARM_AGENT_MODEL_DIR",
    str(REPO.parent / "qwen35_demo" / "models" / "Qwen3.5-9B"),
)


def section(n: int, title: str) -> None:
    print(f"\n{'=' * 76}\n{n}. {title}\n{'=' * 76}")


def main() -> int:
    import torch

    from arm_agent.agent.llm import QwenAgentModel

    print(f"device : {torch.cuda.get_device_name(0)}  "
          f"sm_{''.join(str(x) for x in torch.cuda.get_device_capability(0))}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")
    print(f"torch  : {torch.__version__}  cuda {torch.version.cuda}  "
          f"arch_list {torch.cuda.get_arch_list()}")

    # ---------------------------------------------------------------- 1. backends
    section(1, "is there a fused attention backend on this card?")
    from torch.backends.cuda import can_use_efficient_attention, can_use_flash_attention
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 16, 512, 256, dtype=torch.float16, device="cuda")
    print(f"flash_sdp_enabled    : {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"mem_efficient enabled: "
          f"{torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"math_sdp_enabled     : {torch.backends.cuda.math_sdp_enabled()}")
    for label, fn in (("FLASH_ATTENTION", can_use_flash_attention),
                      ("EFFICIENT_ATTENTION", can_use_efficient_attention)):
        try:
            # params is an SDPAParams-like object in newer torch; the private
            # helper is the documented way to ask this question.
            from torch.backends.cuda import SDPAParams

            p = SDPAParams(q, q, q, None, 0.0, True, False)
            print(f"can_use_{label:<20}: {fn(p)}")
        except Exception as exc:  # noqa: BLE001
            print(f"can_use_{label:<20}: <api changed: {type(exc).__name__}: {exc}>")
    for name, backend in (("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
                          ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
                          ("MATH", SDPBackend.MATH)):
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            print(f"no exception with {name:<20}: YES")
        except Exception as exc:  # noqa: BLE001
            print(f"no exception with {name:<20}: NO ({type(exc).__name__})")
    del q

    # NOTE: "no exception" is NOT proof that a backend ran -- a disabled backend
    # falls back silently. The decisive test is section 4, which measures peak
    # memory per backend: the math path materialises (H, N, N) in fp32 and the
    # fused paths do not, so the two cannot look alike.
    section(2, "decisive: peak memory of one SDPA call, per backend (N=4096)")
    N2, H, D = 4096, 16, 256
    q2 = torch.randn(1, H, N2, D, dtype=torch.float16, device="cuda")
    k2 = torch.randn(1, H, N2, D, dtype=torch.float16, device="cuda")
    v2 = torch.randn(1, H, N2, D, dtype=torch.float16, device="cuda")
    input_b = 3 * q2.numel() * 2
    fp32_scores = H * N2 * N2 * 4
    fp16_scores = H * N2 * N2 * 2
    print(f"inputs (q,k,v fp16)          : {input_b / 2**20:>8.0f} MiB")
    print(f"(H,N,N) fp32 if materialised : {fp32_scores / 2**20:>8.0f} MiB")
    print(f"(H,N,N) fp16 if materialised : {fp16_scores / 2**20:>8.0f} MiB")
    for name, backend in (("MATH", SDPBackend.MATH),
                          ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(q2, k2, v2)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
            print(f"{name:<20} peak_extra {peak / 2**20:>8.0f} MiB  "
                  f"({'materialises' if peak > fp16_scores else 'fused/none'})")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<20} FAILED: {type(exc).__name__}: {exc}")
    del q2, k2, v2
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ 2. weight map
    section(3, "where do the 7.34 GiB of weights go?")
    model = QwenAgentModel(MODEL_DIR, load="4bit", gpu_mem_gib=9.5,
                           image_max_side=160, temperature=0.0)
    model.load_model()
    torch.cuda.synchronize()
    total = torch.cuda.memory_allocated()

    from collections import defaultdict

    by_dtype: dict[str, int] = defaultdict(int)
    by_prefix: dict[str, int] = defaultdict(int)
    for name, p in model._model.named_parameters():
        n = p.numel() * p.element_size()
        by_dtype[str(p.dtype)] += n
        by_prefix[".".join(name.split(".")[:3])] += n
    for name, b in model._model.named_buffers():
        if b is not None and b.numel():
            by_prefix["BUFFER:" + ".".join(name.split(".")[:2])] += \
                b.numel() * b.element_size()

    print(f"allocated after load : {total / 2**30:.2f} GiB")
    print("\nby dtype:")
    for d, n in sorted(by_dtype.items(), key=lambda kv: -kv[1]):
        print(f"  {d:<16} {n / 2**30:>6.3f} GiB")
    print("\ntop 12 groups by bytes (module path prefix):")
    for k, n in sorted(by_prefix.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:<44} {n / 2**30:>6.3f} GiB")

    # ------------------------------------------------------------------ 3. prefix
    section(4, "how much does PREFIX REUSE save?")
    from arm_agent.agent import prompts as P

    def render(n_pairs: int):
        from PIL import Image

        msgs = [{"role": "system", "content": [{"type": "text", "text": P.SYSTEM_PROMPT_PLAIN}]},
                {"role": "user", "content": [{"type": "text", "text":
                 "任务：pick up the alphabet soup and place it in the basket。\n"
                 "[上图是全局相机看到的场景]\n请开始。记住：每次只调用一个工具。"}]}]
        call = ("<tool_call>\n<function=locate>\n<parameter=name>\n"
                "alphabet_soup\n</parameter>\n</function>\n</tool_call>")
        reply = ("[locate] alphabet_soup_1: (-0.1191, -0.2398, +0.0384) m\n"
                 "  机械爪当前: (-0.1522, -0.0066, +0.2487) m\n"
                 "  偏差（物体 - 机械爪）: dx=+3.3cm, dy=-23.3cm, dz=-21.0cm")
        for i in range(n_pairs):
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": call}]})
            msgs.append({"role": "user", "content": [{"type": "text", "text": reply}]})
        text = model._tokenizer.apply_chat_template(
            msgs, tools=P.TOOL_SCHEMAS, tokenize=False,
            enable_thinking=False, add_generation_prompt=True)
        return model._tokenizer(text, return_tensors="pt")

    # 11 pairs ~= 3.57k tokens, below the measured ceiling (~3.9k).
    # Two earlier attempts failed here for the same reason: 26 pairs = 5599 tok
    # OOMed mid-section, and 14 pairs = 3979 tok tripped this guard. Using a
    # prompt size the other probe had ALREADY measured to be impossible is not a
    # measurement, it is a bug -- hence the guard above and this margin.
    N_PAIRS = 11
    enc = render(N_PAIRS)
    ids = enc["input_ids"]
    N = int(ids.shape[1])
    print(f"full prompt : {N} tokens ({N_PAIRS} pairs)  "
          f"[card ceiling measured at ~3.9k]")
    if N > 3850:
        print(f"[FAIL] prompt {N} is above the measured ceiling; lower N_PAIRS")
        return 1
    dev = model._model.device

    def timed(fn) -> float:
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        return time.time() - t0

    n_new = 180  # one more history pair, i.e. what the loop actually appends
    head = ids[:, : N - n_new].to(dev)
    tail = ids[:, N - n_new:].to(dev)

    # A) cold: prefill the whole thing every turn (what the harness does today)
    def cold() -> None:
        with torch.inference_mode():
            model._model(input_ids=ids.to(dev), use_cache=False)

    # B) warm: prefill the head ONCE, then only the 180 new tokens
    t_head = timed(lambda: model._model(input_ids=head, use_cache=True))
    cache = None
    with torch.inference_mode():
        out = model._model(input_ids=head, use_cache=True)
        cache = out.past_key_values

    def warm() -> None:
        with torch.inference_mode():
            model._model(input_ids=tail, past_key_values=cache, use_cache=True)

    # C) upper bound for incremental-only: prefill 180 tokens with no history
    def tiny() -> None:
        with torch.inference_mode():
            model._model(input_ids=tail, use_cache=False)

    t_cold = min(timed(cold) for _ in range(2))
    t_warm = min(timed(warm) for _ in range(2))
    t_tiny = min(timed(tiny) for _ in range(2))
    print(f"\nA) cold full prefill ({N} tok, every turn)      : {t_cold * 1000:>8.0f} ms")
    print(f"   (one-off head prefill for the cache, {N - n_new} tok): {t_head * 1000:>8.0f} ms")
    print(f"B) warm incremental (+{n_new} tok, cache reused) : {t_warm * 1000:>8.0f} ms")
    print(f"C) {n_new} tok with no history at all           : {t_tiny * 1000:>8.0f} ms")
    if t_warm > 0:
        print(f"   -> warm/cold = {t_warm / t_cold:.3f}  "
              f"(1 - ratio) = {1 - t_warm / t_cold:.0%} of the per-turn prefill removed")

    print("\nself-checks:")
    bad = 0
    if t_warm < t_cold:
        print("[PASS] prefix reuse is faster than a cold full prefill")
    else:
        print("[FAIL] reuse was not faster -> the cache is not being used")
        bad += 1
    if t_tiny <= t_cold:
        print("[PASS] cold prefill of the full prompt costs more than 180 tokens alone")
    else:
        print("[FAIL] sanity check on timing failed")
        bad += 1

    print(f"\n{bad} self-check failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
