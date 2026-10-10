"""Does a long context make the model release where a short one would not?

The single-turn probes showed the model judges "the basket is not under the gripper"
correctly from the same frames -- so the failure was attributed to the judging step
being skipped.  The counter-hypothesis is that the CONTEXT is what breaks it: with
8 turns of its own confident prose in the window, it may talk itself into "arrived".

This rebuilds the exact harness conversation up to the release decision of a real
run (the `gripper` call immediately before the first `verify`), then re-asks for the
next action with progressively shorter history: full / last 6 / last 3 / last 1 turn.
Everything else is held fixed -- same system prompt, same tool schemas, same images
on the newest two messages, same `prepare_for_model` elision, greedy decoding.

If history is the cause, the short variants should choose to keep moving while the
full one releases.  If all four release, the context is not the deciding factor.

Usage:
    ARM_AGENT_MODEL_DIR=/dev/shm/arm_agent_model PYTHONNOUSERSITE=1 PYTHONPATH=src \
      /root/miniconda3/envs/qwen35/bin/python scripts/probe_context_length.py \
      [--episode outputs/episodes/episode_000_1791626311.json] [--turns 8]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from PIL import Image  # noqa: E402

AGENT_BOX = (160, 20, 480, 340)
WRIST_BOX = (160, 380, 480, 700)
SOURCE_SIDE = 256


def _load_sibling():
    """Reuse the release-probe helpers (frame text / state-line reconstruction)."""
    spec = importlib.util.spec_from_file_location(
        "probe_release_perception", REPO / "scripts" / "probe_release_perception.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _calls(transcript: list[dict]) -> list[tuple[int, str, dict[str, str]]]:
    """(turn, tool, params) for every assistant reply that carries a tool call.

    Parsed from the reply text, not from `structured_result`: the `verify` turns
    were stored WITHOUT one (only `result` + `success`), so a structured lookup
    silently finds "no verify call in this episode".
    """
    out: list[tuple[int, str, dict[str, str]]] = []
    for e in transcript:
        if "assistant" not in e:
            continue
        m = re.search(r"<function=(\w+)>", e["assistant"])
        if not m:
            continue
        params = dict(re.findall(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", e["assistant"], re.S))
        out.append((int(e["turn"]), m.group(1), params))
    return out


def release_decision_turn(transcript: list[dict]) -> tuple[int, int]:
    """(first verify turn, the gripper open right before it)."""
    calls = _calls(transcript)
    verify_turn = next((t for t, name, _ in calls if name == "verify"), None)
    if verify_turn is None:
        raise SystemExit("no verify call in this episode")
    gripper_turn = None
    for t, name, params in calls:
        if t >= verify_turn:
            continue
        if name == "gripper" and params.get("action", "").strip() == "open":
            gripper_turn = t
    if gripper_turn is None:
        raise SystemExit("no gripper open before the verify call")
    return verify_turn, gripper_turn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="outputs/episodes/episode_000_1791626311.json")
    ap.add_argument("--video", default="outputs/videos/ep000_1791625655/video.mp4")
    ap.add_argument("--load", default=os.environ.get("LOAD", "bf16"))
    ap.add_argument("--gpu-mem-gib", type=float, default=float(os.environ.get("GPU_MEM_GIB", 29)))
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--image-keep", type=int, default=2, help="harness HISTORY_IMAGE_KEEP")
    # --ask: instead of letting the model choose an action, append the release
    # question to the last turn and drop the tool schemas -- this is exactly what
    # the harness-side gate (A') would do, tested at the REAL context length.
    ap.add_argument("--ask", action="store_true")
    # --thinking: turn on the chat template's reasoning block (llm.enable_thinking).
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    episode_path = str(REPO / args.episode) if not Path(args.episode).is_absolute() else args.episode
    prp = _load_sibling()
    prp.EPISODE = episode_path  # the helpers read this global
    d = json.load(open(episode_path))
    transcript = d["transcript"]

    verify_turn, release_turn = release_decision_turn(transcript)
    prefix_turns = list(range(1, release_turn))  # turns 1..release_turn-1 are history
    print(f"episode: {args.episode}")
    print(f"第一次 verify = t{verify_turn}；它之前那次 gripper open = t{release_turn}")
    print(f"→ 重现 t{release_turn} 的决策：上下文取 t1..t{release_turn - 1} 的历史\n")

    # ---- frames ----------------------------------------------------------
    import imageio.v3 as iio

    need = set(prefix_turns) | {0}
    frames: dict[int, tuple[Image.Image, Image.Image]] = {}
    for i, arr in enumerate(iio.imiter(str(REPO / args.video))):
        if i in need:
            img = Image.fromarray(arr)
            frames[i] = (
                img.crop(AGENT_BOX).resize((SOURCE_SIDE, SOURCE_SIDE), Image.LANCZOS),
                img.crop(WRIST_BOX).resize((SOURCE_SIDE, SOURCE_SIDE), Image.LANCZOS),
            )
        if i > max(need):
            break
    missing = [t for t in need if t not in frames]
    if missing:
        raise SystemExit(f"video missing frames {missing}")

    grasped = prp.grasp_timeline()

    def imgs(turn: int) -> list[dict]:
        agent, wrist = frames[turn]
        return [{"type": "image", "image": agent}, {"type": "image", "image": wrist}]

    # First user message: the exact text is not stored, so rebuild it. Its state
    # line is taken from turn 1's measurement (the closest stored one); with the
    # full history in the window it carries no weight either way.
    language = d["result"].get("language") or d["result"].get("task")
    first_user = (
        f"任务：{language}\n"
        f"[上图：全局相机 + 腕部相机（从上往下看夹爪下方，用来判断自己在哪）]\n"
        f"机械爪位置 (-0.1527, -0.0062, 0.2483) m；夹爪：张开；爪上：空（没有物体跟着手）\n"
        f"请开始。记住：每次只调用一个工具。"
    )

    def build(upto_turn: int | None) -> list[dict]:
        """Rebuild the harness message list by walking the transcript in order.

        Three entry kinds exist and each maps to a different message:
          assistant   -> assistant text (verbatim)
          result      -> user message: two cameras + result text + json + format_state
          tool_error  -> user message: TEXT ONLY, no images (harness line 266)
          protocol_retry -> user message: TEXT ONLY

        Skipping the tool_error entry (as a per-turn rebuild does) changes both the
        message COUNT and the content of that turn, which is enough to move the
        model off the action the loop actually took -- found the hard way on turn 3
        of episode_000_1791626311, where `approach 'blue can'` raised.
        """
        msgs: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": [*imgs(0), {"type": "text", "text": first_user}]},
        ]
        for e in transcript:
            t = e.get("turn")
            if t is None or (upto_turn is not None and t > upto_turn):
                continue
            if "assistant" in e:
                msgs.append({"role": "assistant", "content": e["assistant"]})
            elif "result" in e:
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            *imgs(t),
                            {"type": "text", "text": prp.reconstructed_message(t, grasped.get(t, False))},
                        ],
                    }
                )
            elif "tool_error" in e:
                msgs.append({"role": "user", "content": e["tool_error"]})
            elif e.get("protocol_retry"):
                msgs.append(
                    {"role": "user", "content": "工具协议错误。请只输出一个合法工具调用，并修正参数。"}
                )
        return msgs

    variants: list[tuple[str, int | None, list[int] | None]] = [
        (f"full  (t1..t{release_turn - 1})", release_turn - 1, None),
        ("last6", None, prefix_turns[-6:]),
        ("last3", None, prefix_turns[-3:]),
        ("last1", None, prefix_turns[-1:]),
    ]

    from arm_agent.agent.llm import QwenAgentModel, prepare_for_model
    from arm_agent.agent.prompts import get_system_prompt, get_tool_schemas

    system = get_system_prompt("v2")
    tools = get_tool_schemas("v2")

    model = QwenAgentModel(
        model_dir=os.environ.get(
            "ARM_AGENT_MODEL_DIR", str(REPO.parent / "qwen35_demo" / "models" / "Qwen3.5-9B")
        ),
        load=args.load,
        gpu_mem_gib=args.gpu_mem_gib,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        image_max_side=160,
        enable_thinking=args.thinking,
    )
    model.load_model()

    def build_suffix(keep_turns: list[int]) -> list[dict]:
        """System + task message + only the entries belonging to `keep_turns`."""
        keep = set(keep_turns)
        rebuilt: list[dict] = build(0)[:2]  # system + user0 (turn filter 0 keeps neither)
        for e in transcript:
            t = e.get("turn")
            if t is None or t not in keep:
                continue
            if "assistant" in e:
                rebuilt.append({"role": "assistant", "content": e["assistant"]})
            elif "result" in e:
                rebuilt.append(
                    {
                        "role": "user",
                        "content": [
                            *imgs(t),
                            {"type": "text", "text": prp.reconstructed_message(t, grasped.get(t, False))},
                        ],
                    }
                )
            elif "tool_error" in e:
                rebuilt.append({"role": "user", "content": e["tool_error"]})
        return rebuilt

    for label, upto, keep_turns in variants:
        messages = build(upto) if keep_turns is None else build_suffix(keep_turns)
        if args.ask:
            # Append the question to the last user message's text block.
            last = messages[-1]
            assert last["role"] == "user" and isinstance(last["content"], list)
            last["content"][-1] = {
                "type": "text",
                "text": last["content"][-1]["text"]
                + "\n\n"
                + "只根据画面回答，不要调用任何工具：\n"
                + "1. 夹爪正下方现在是什么？\n"
                + "2. 如果此刻张开夹爪，手里的物体会落到哪里？会不会进入画面里那个容器？\n"
                + "3. 先说结论：现在放手可以吗？（可以 / 不可以）",
            }
            label += " +ask"
        prepared = prepare_for_model(messages, args.image_keep)
        n_img = sum(
            1
            for m in prepared
            if isinstance(m.get("content"), list)
            for it in m["content"]
            if isinstance(it, dict) and it.get("type") == "image"
        )
        reply = model.chat(prepared, None if args.ask else tools)
        m = re.search(r"<function=(\w+)>", reply)
        params = dict(re.findall(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", reply, re.S))
        action = f"{m.group(1)} {params}" if m else "(无 XML 工具调用)"
        print("=" * 100)
        print(f"[{label}]  消息 {len(prepared)} 条 / 保留图像 {n_img} 张")
        print(f"  循环里 t{release_turn} 实际动作: {prp.loop_action(release_turn)}")
        print(f"  → 本次选择: {action}")
        print("-" * 100)
        print(reply.strip())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
