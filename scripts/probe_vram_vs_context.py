#!/usr/bin/env python3
"""Measure REAL peak VRAM as the context grows, then extrapolate the ceiling.

WHY THIS EXISTS
---------------
The M1 loop runs at `MAX_HISTORY_MESSAGES = 4` / `HISTORY_IMAGE_KEEP = 1`,
i.e. ~2.4-3.0k prompt tokens. That is a memory decision, not a design one, and
it is the reason the model forgets the previous stage of the task: the causal
chain "descend onto the payload I am holding -> squeeze -> drop" spans t5..t7
and simply is not in the window.

Every existing probe on this subject (`probe_token_budget2.py`,
`bench_token_budget.py`) counts TOKENS on the CPU and applies the analytic
formula `N^2 * heads * 2 bytes`. That formula is what `llm.py` prints as
`attn_matrix≈`, and it is a PREDICTION, never a measurement -- fitting it back to
itself proves nothing. What is missing is the measured curve of
`torch.cuda.max_memory_allocated()` against N, which is what decides where the
next OOM is.

WHAT IT MEASURES
  1. the peak VRAM of one `generate()` at increasing prompt lengths, with the
     real system prompt + the real 11 tool schemas, until it OOMs;
  2. the marginal cost of one extra history message, and of one extra image, so
     "how many past steps can I keep" becomes a number;
  3. the fitted growth law, checked against the model config.

SELF-CHECKS (a probe that cannot contradict itself is just a nice table)
  * token count must be strictly monotone in the requested size -- otherwise the
    x-axis is meaningless;
  * the measured peak must strictly increase -- otherwise `reset_peak_memory_stats`
    is not doing what we think and every number below is noise;
  * the growth must be QUADRATIC in N. If it comes out linear, the N^2
    extrapolation this probe exists to produce is invalid, and the probe says so
    instead of printing a confident table.

Run (GPU node, inside the qwen35 env):
    bash scripts/run_vram_probe_on_gpu.sh
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_DIR = os.environ.get(
    "ARM_AGENT_MODEL_DIR",
    str(REPO.parent / "qwen35_demo" / "models" / "Qwen3.5-9B"),
)


def mib(b: float) -> float:
    return b / 2 ** 20


def gib(b: float) -> float:
    return b / 2 ** 30


# --------------------------------------------------------------------- prompts
def make_messages(n_pairs: int, n_images: int, system_prompt: str, tools):
    """A synthetic conversation shaped like a real episode.

    One `pair` == one assistant tool call + the user-role tool result, i.e. two
    history messages, exactly the unit `MAX_HISTORY_MESSAGES` counts.
    """
    from PIL import Image

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text":
            "任务：pick up the alphabet soup and place it in the basket。\n"
            "[上图是全局相机看到的场景]\n请开始。记住：每次只调用一个工具。"}]},
    ]
    call = ("<tool_call>\n<function=locate>\n<parameter=name>\n"
            "alphabet_soup\n</parameter>\n</function>\n</tool_call>")
    reply = (
        "[locate] alphabet_soup_1: (-0.1191, -0.2398, +0.0384) m\n"
        "  机械爪当前: (-0.1522, -0.0066, +0.2487) m\n"
        "  偏差（物体 - 机械爪）: dx=+3.3cm, dy=-23.3cm, dz=-21.0cm\n"
        "机械爪位置 (-0.1522, -0.0066, +0.2487) m；夹爪：闭合；爪上：空（没有物体跟着手）"
    )
    for i in range(n_pairs):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": call}]})
        content = [{"type": "text", "text": reply}]
        if i >= n_pairs - n_images:
            content.append({"type": "image", "image": Image.new("RGB", (256, 256))})
        messages.append({"role": "user", "content": content})
    return messages


def render(model, messages, tools, image_side: int):
    from arm_agent.agent.llm import _collect_images  # noqa: PLC0415

    images = _collect_images(messages, max_side=image_side)
    text = model._tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False,
        enable_thinking=False, add_generation_prompt=True,
    )
    pk = {}
    if os.environ.get("IMAGE_MIN_PIXELS"):
        pk["size"] = {"shortest_edge": int(os.environ["IMAGE_MIN_PIXELS"]),
                      "longest_edge": 262144}
    if images and model._processor is not None:
        enc = model._processor(text=[text], images=images, return_tensors="pt", **pk)
    else:
        enc = model._tokenizer(text, return_tensors="pt")
    return enc, len(images)


def pairs_for_target(model, tools, system_prompt, target: int, image_side: int) -> int:
    """Smallest pair count whose rendered prompt reaches `target` tokens.

    Token count is monotone in the pair count, so a doubling scan + a short
    linear walk is enough and avoids rendering a 90-pair template 90 times.
    """
    lo, hi = 0, 1
    while True:
        n = len(render(model, make_messages(hi, 0, system_prompt, tools), tools, image_side)[0]["input_ids"][0])
        if n >= target or hi > 512:
            break
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        n = len(render(model, make_messages(mid, 0, system_prompt, tools), tools, image_side)[0]["input_ids"][0])
        if n >= target:
            hi = mid
        else:
            lo = mid
    return hi


def one_generate(model, enc, max_new_tokens: int = 8) -> float:
    """Run one generate() and return its wall time in seconds.

    Time is measured here, not just memory: the same N^2 that drives the peak
    allocation also drives the prefill FLOPs, and on a Turing card the compute
    term may become the binding constraint BEFORE memory does. A memory-only
    table would silently recommend a context the loop cannot afford in latency.
    """
    import time as _time

    import torch

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inputs = {k: v.to(model._model.device) for k, v in enc.items() if hasattr(v, "to")}
    t0 = _time.time()
    with torch.inference_mode():
        model._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=model._tokenizer.eos_token_id,
            pad_token_id=model._tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    dt = _time.time() - t0
    del inputs
    return dt


def main() -> int:
    import torch

    from arm_agent.agent.llm import QwenAgentModel
    from arm_agent.agent.prompts import SYSTEM_PROMPT_PLAIN, TOOL_SCHEMAS

    assert torch.cuda.is_available(), "needs a GPU"
    total_b = torch.cuda.get_device_properties(0).total_memory
    print("=" * 78)
    print("VRAM vs CONTEXT LENGTH -- measured, not modelled")
    print("=" * 78)
    print(f"device          : {torch.cuda.get_device_name(0)}")
    print(f"total VRAM      : {gib(total_b):.2f} GiB")
    caps = torch.cuda.get_device_capability(0)
    print(f"compute cap     : sm_{caps[0]}{caps[1]}")
    print(f"model dir       : {MODEL_DIR}")

    from arm_agent.agent import prompts as P

    image_side = int(os.environ.get("IMAGE_SIDE", "160"))
    attn_impl = os.environ.get("ATTN_IMPL", "")  # '', 'eager', 'sdpa'

    # FORCE_EFFICIENT_ATTN=1 turns off the math and flash SDPA backends globally,
    # leaving only the memory-efficient one.
    #
    # WHY: probe_arch_levers measured that a single SDPA call at N=4096, 16 heads
    # costs 2560 MiB on the math path (= fp32 scores 1024 + fp32 softmax output
    # 1024 + fp16 cast 512, exactly) and 96 MiB on the memory-efficient path (just
    # the inputs). 2560 MiB at 4096 tokens is ~160 B/tok^2 -- the measured law of
    # THIS probe. So the whole context ceiling is set by which backend SDPA
    # silently picks, and on sm_75 it picks math even though the efficient
    # backend is available and works. This knob tests whether pinning the backend
    # really moves B, which is the difference between a 5x context win and a
    # nice-sounding theory.
    if os.environ.get("FORCE_EFFICIENT_ATTN") == "1":
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("backends        : math OFF, flash OFF, mem_efficient ON "
              "(FORCE_EFFICIENT_ATTN=1)")
    else:
        print(f"backends        : math {torch.backends.cuda.math_sdp_enabled()}, "
              f"flash {torch.backends.cuda.flash_sdp_enabled()}, "
              f"mem_efficient {torch.backends.cuda.mem_efficient_sdp_enabled()}")

    # GQA_EXPAND=1: expand K/V to the query head count before SDPA.
    #
    # MEASURED (probe_arch_levers + a direct kernel test), and this supersedes my
    # earlier "sm_75 has no fused kernel" explanation:
    #   * flash attention needs sm80+          -> unavailable here, permanently;
    #   * memory-efficient attention WORKS on   sm_75 (96 MiB for a 4096-token,
    #     16-head call) -- but BOTH fused kernels refuse GQA: transformers passes
    #     q with 16 heads and k/v with 4 (num_key_value_heads=4), and the dense
    #     kernels require equal head counts. So SDPA silently falls to the MATH
    #     path, which materialises fp32 scores + fp32 softmax out + the fp16 cast
    #     = 2560 MiB at N=4096, i.e. the 170.7 B/tok^2 this probe measures.
    #   * `enable_gqa=True` does NOT help on this card (still "No available
    #     kernel"), but expanding K/V by hand DOES: 2560 MiB -> 96 MiB, 27x.
    # The expansion multiplies the KV cache by 4 (32 -> 128 KiB/token), which is
    # cheap next to the N^2 term it removes.
    if os.environ.get("GQA_EXPAND") == "1":
        _orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def _sdpa_gqa_expand(q, k, v, *a, **kw):
            if (q.dim() == 4 and k.dim() == 4 and k.shape[1] != q.shape[1]
                    and q.shape[1] % k.shape[1] == 0):
                rep = q.shape[1] // k.shape[1]
                k = k.unsqueeze(2).expand(-1, k.shape[1], rep, -1, -1).reshape(
                    q.shape[0], q.shape[1], k.shape[2], k.shape[3])
                v = v.unsqueeze(2).expand(-1, v.shape[1], rep, -1, -1).reshape(
                    q.shape[0], q.shape[1], v.shape[2], v.shape[3])
                # CRUCIAL: drop enable_gqa once K/V are already expanded.
                # transformers/integrations/sdpa_attention.py sets
                # sdpa_kwargs = {"enable_gqa": True} whenever q and k have
                # different head counts, and a kernel test measured that
                # enable_gqa=True is itself enough to make the fused kernel refuse
                # ("No available kernel") even on equal head counts. The first
                # version of this wrapper forwarded it, so the expansion had NO
                # effect end-to-end (B stayed at 188-212 tok^2) -- a wrapper that
                # silently keeps the disqualifying flag is indistinguishable from
                # no wrapper at all.
                kw.pop("enable_gqa", None)
            return _orig_sdpa(q, k, v, *a, **kw)

        torch.nn.functional.scaled_dot_product_attention = _sdpa_gqa_expand
        print("GQA expand      : ON (K/V repeated up to the query head count)")

    model = QwenAgentModel(MODEL_DIR, load="4bit", gpu_mem_gib=9.5,
                           image_max_side=image_side, temperature=0.0,
                           attn_impl=attn_impl or None)
    model.load_model()
    # Which attention path is actually running decides whether the N x N matrix
    # is materialised at all, so print it rather than assuming.
    print(f"attn impl       : "
          f"{getattr(model._model.config, '_attn_implementation', '?')} "
          f"(requested {attn_impl or 'default'})")
    print(f"image_max_side  : {image_side}px  "
          f"(processor shortest_edge="
          f"{getattr(model._processor.image_processor.size, 'shortest_edge', '?') if model._processor is not None else 'n/a'})")

    torch.cuda.synchronize()
    weights_b = torch.cuda.memory_allocated()
    print(f"\nweights after load : {gib(weights_b):.2f} GiB allocated "
          f"({gib(total_b - weights_b):.2f} GiB left before anything else)")

    # A prompt is rendered once to learn the tokens-per-pair slope; that slope is
    # what turns a token budget into "how many past steps fit".
    e6, _ = render(model, make_messages(6, 0, P.SYSTEM_PROMPT_PLAIN, TOOL_SCHEMAS),
                   TOOL_SCHEMAS, image_side)
    e12, _ = render(model, make_messages(12, 0, P.SYSTEM_PROMPT_PLAIN, TOOL_SCHEMAS),
                    TOOL_SCHEMAS, image_side)
    n6 = int(e6["input_ids"].shape[1])
    n12 = int(e12["input_ids"].shape[1])
    per_pair = (n12 - n6) / 6.0
    print(f"\nper history pair   : {per_pair:.1f} tok/pair "
          f"(1 pair = 1 assistant call + 1 tool result = 2 messages)")
    print(f"fixed overhead     : {n6 - 6 * per_pair:.0f} tok "
          f"(system + 11 tool schemas + first user turn)")

    # ---------------------------------------------------------------- 1. sweep
    # Dense and LOW on purpose: the first version of this probe jumped 2500 ->
    # 5000 and the 5000 point OOMed, leaving ONE data point and therefore no
    # law to extrapolate -- while still printing "0 self-check failure(s)".
    targets = [int(x) for x in os.environ.get(
        "TARGETS", "1200,1700,2200,2700,3200,3700,4200"
    ).split(",")]
    print("\n--- sweep: peak VRAM of one generate() vs prompt length ---")
    print(f"{'target':>8} {'pairs':>6} {'ctx_tok':>9} {'peak_alloc':>11} "
          f"{'peak-weights':>13} {'N^2 law':>9} {'free_after':>11} {'gen_s':>7}  note")
    print("-" * 100)

    rows: list[tuple[int, float, float]] = []  # (N, peak_bytes, peak_minus_weights)
    times: list[tuple[int, float]] = []
    prev_cap = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for target in targets:
        n_pairs = pairs_for_target(model, TOOL_SCHEMAS, P.SYSTEM_PROMPT_PLAIN,
                                   target, image_side)
        enc, n_img = render(model, make_messages(n_pairs, 0, P.SYSTEM_PROMPT_PLAIN,
                                                TOOL_SCHEMAS), TOOL_SCHEMAS, image_side)
        n_tok = int(enc["input_ids"].shape[1])
        predicted = n_tok * n_tok * 16 * 2  # the analytic form llm.py prints
        try:
            dt = one_generate(model, enc)
            peak = torch.cuda.max_memory_allocated()
            free_b, _ = torch.cuda.mem_get_info()
            rows.append((n_tok, float(peak), float(peak - weights_b)))
            times.append((n_tok, dt))
            print(f"{target:>8} {n_pairs:>6} {n_tok:>9} {gib(peak):>9.2f}GiB "
                  f"{mib(peak - weights_b):>11.0f}MiB {mib(predicted):>7.0f}MiB "
                  f"{gib(free_b):>9.2f}GiB {dt:>7.2f}")
        except torch.cuda.OutOfMemoryError as exc:
            free_b, _ = torch.cuda.mem_get_info()
            # Keep the requested size: it is the actual size of the block that
            # could not be served, i.e. the real granularity of the ceiling.
            req = ""
            try:
                import re as _re

                m = _re.search(r"tried to allocate ([0-9.]+) (MiB|GiB)", str(exc))
                req = f"need {m.group(1)}{m.group(2)}" if m else ""
            except Exception:
                pass
            print(f"{target:>8} {n_pairs:>6} {n_tok:>9} {'OOM':>11} {'':>13} "
                  f"{mib(predicted):>7.0f}MiB {gib(free_b):>9.2f}GiB {'':>7}  "
                  f"<-- CEILING {req}")
            torch.cuda.empty_cache()
            break
        finally:
            del enc
            torch.cuda.empty_cache()

    # -------------------------------------------------------- 2. self-checks
    print("\n--- self-checks ---")
    bad = 0
    inconclusive = len(rows) < 3
    if inconclusive:
        # Do NOT print "0 failure(s)" here. The first version did, after the
        # sweep died at the second point and every check below was skipped --
        # a green summary over a table with no law in it is exactly the failure
        # mode this file's docstring warns about.
        print(f"[INCONCLUSIVE] only {len(rows)} successful point(s); "
              f"need >= 3 to fit any law. Nothing below was validated.")
        bad += 1
    if len(rows) >= 2:
        ns = [r[0] for r in rows]
        if all(b > a for a, b in zip(ns, ns[1:])):
            print("[PASS] token count is strictly monotone in the target")
        else:
            print("[FAIL] token count did not grow -> the x-axis is meaningless")
            bad += 1
        peaks = [r[1] for r in rows]
        if all(b > a for a, b in zip(peaks, peaks[1:])):
            print("[PASS] measured peak grows with prompt length")
        else:
            print("[FAIL] peak did not grow -> reset_peak_memory_stats is not working")
            bad += 1
        # Quadratic or not? Compare the growth of (peak - weights) against N^2.
        inc = [r[2] for r in rows]
        ratio = [inc[i] / (ns[i] ** 2) for i in range(len(rows))]
        print(f"       (peak-weights)/N^2 = "
              f"{', '.join(f'{r*1e6:.3f}' for r in ratio)} B/tok^2  "
              f"(predict {16 * 2} B/tok^2 for one fp16 16-head matrix)")
        core = ratio[len(ratio) // 2:]
        spread = (max(core) - min(core)) / max(min(core), 1e-12) if core else 1.0
        if spread < 0.35:
            print(f"[PASS] growth is quadratic: B/tok^2 varies {spread*100:.0f}% "
                  f"across the upper half of the sweep")
        else:
            print(f"[FAIL] growth is NOT quadratic (B/tok^2 varies "
                  f"{spread*100:.0f}%) -> do not extrapolate from N^2")
            bad += 1
        # -------------------------------------------------- 3. extrapolation
        if bad == 0:
            # TWO-parameter fit: peak_extra(N) = c + b*N^2.
            #
            # A one-parameter fit (b = peak/N^2 averaged) is what this file did
            # first, and it is wrong in a way that matters: the measured
            # B/tok^2 falls from 214 to 186 across the sweep, which is the
            # signature of a POSITIVE CONSTANT ~250 MiB (the vision tower's
            # buffers and the first-touch allocations), not of a smaller
            # quadratic term. Ignoring the constant makes the extrapolation
            # over-optimistic at large N.
            def lsq(xs: list[float], ys: list[float]) -> tuple[float, float]:
                n = len(xs)
                sx, sy = sum(xs), sum(ys)
                sxx = sum(x * x for x in xs)
                sxy = sum(x * y for x, y in zip(xs, ys))
                den = n * sxx - sx * sx
                if den == 0:
                    return 0.0, sy / n
                slope = (n * sxy - sx * sy) / den
                return slope, (sy - slope * sx) / n

            xs = [float(n) ** 2 for n, _, _ in rows]
            b, c = lsq(xs, [inc for _, _, inc in rows])
            print("\n--- extrapolation (two-parameter fit on the measured points) ---")
            print(f"       peak_extra(N) = {c / 2**20:.0f} MiB + {b:.1f} B * N^2")
            print(f"       llm.py's analytic 32 B/tok^2 is {32 / b:.2f}x SMALLER "
                  f"than measured")

            # Time is fitted the same way, on the SAME rows. Attributing all of
            # gen_s to N^2 (t = t_unit*N^2) over-estimates badly here because the
            # measured time is nearly flat (2.47 -> 2.94 s for a 2.76x rise in
            # N^2) -- i.e. it is dominated by a constant, and the constant is
            # mostly fixed launch/decoding cost.
            if len(times) >= 2:
                t2, t0 = lsq([float(n) ** 2 for n, _ in times],
                             [t for _, t in times])
            else:
                t2 = t0 = 0.0
            print(f"       gen_s(N)      = {t0:.2f} s + {t2 * 1e9:.2f}e-9 s * N^2")

            margin = 0.6 * 2 ** 30
            print(f"       margin reserved for vision + KV + fragmentation: "
                  f"{margin / 2**30:.1f} GiB")
            print(f"{'VRAM':>8} {'usable':>9} {'N ceiling':>11} {'history pairs':>14} "
                  f"{'= messages':>11} {'gen_s est':>10}")
            for cap_gib in (10.57, 16, 24, 32, 48, 80):
                cap = cap_gib * 2 ** 30
                usable = cap - weights_b - margin
                n_max = (max(usable - c, 0.0) / b) ** 0.5
                pairs = max(0.0, (n_max - (n6 - 6 * per_pair)) / per_pair)
                print(f"{cap_gib:>6.2f}GiB {gib(usable):>7.2f}GiB "
                      f"{n_max:>11.0f} {pairs:>14.0f} {pairs * 2:>11.0f} "
                      f"{t0 + t2 * n_max ** 2:>9.1f}s")

    # ------------------------------------------------- 4. marginal costs
    print("\n--- marginal cost of one extra history message / image ---")
    base_pairs = 6
    e_a, _ = render(model, make_messages(base_pairs, 0, P.SYSTEM_PROMPT_PLAIN,
                                         TOOL_SCHEMAS), TOOL_SCHEMAS, image_side)
    e_b, _ = render(model, make_messages(base_pairs + 1, 0, P.SYSTEM_PROMPT_PLAIN,
                                         TOOL_SCHEMAS), TOOL_SCHEMAS, image_side)
    print(f"  + 2 messages (1 call + 1 result) : "
          f"{int(e_b['input_ids'].shape[1]) - int(e_a['input_ids'].shape[1])} tok")
    e_c, n_img = render(model, make_messages(base_pairs, 1, P.SYSTEM_PROMPT_PLAIN,
                                             TOOL_SCHEMAS), TOOL_SCHEMAS, image_side)
    print(f"  + 1 image at {image_side}px           : "
          f"{int(e_c['input_ids'].shape[1]) - int(e_a['input_ids'].shape[1])} tok "
          f"({n_img} image attached)")
    for side in (160, 256, 384):
        e_raw, _ = render(model, make_messages(base_pairs, 1, P.SYSTEM_PROMPT_PLAIN,
                                               TOOL_SCHEMAS), TOOL_SCHEMAS, side)
        # how many image tokens at this resolution: differencing against no image
        print(f"  image tokens @ {side}px max_side    : "
              f"{int(e_raw['input_ids'].shape[1]) - int(e_a['input_ids'].shape[1])} tok")

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
