"""Ask the model to judge the release point from a SINGLE turn of real frames.

Why: the 80-turn run asserted "夹爪已到达篮子正上方" five times (t8/t15/t21/t26/t31)
while the measured EEF was 5.3-7.4 cm OUTSIDE the basket opening in x every time
and up to 29 cm short in y.  That is either (a) the model cannot see the relation
from the pixels it is given, or (b) it can see it and the multi-turn context /
system prompt talks it out of it.

This probe isolates (b) from (a): one user message = the two real frames of that
turn (agentview + wrist, exactly the pipeline the harness uses) + one neutral
text question about the current state.  No tools, no history, no tool results.

Ground truth (mesh-accurate opening, libero_object task 0, init_state 0):
    opening x in [-0.0665, +0.0957]   y in [+0.1661, +0.3395]
    centre  (0.0146, 0.2528)

Usage:
    ARM_AGENT_MODEL_DIR=/dev/shm/arm_agent_model bash scripts/simenv.sh \
        scripts/probe_release_perception.py [--condition neutral|v2|both]
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from PIL import Image  # noqa: E402

from arm_agent.agent.llm import QwenAgentModel  # noqa: E402

VIDEO = REPO / "outputs" / "videos" / "ep000_1791622918" / "video.mp4"
EPISODE = REPO / "outputs" / "episodes" / "episode_000_1791623248.json"

# Composite layout from agent/render.py: two 320x320 panels inside the left 640px.
AGENT_BOX = (160, 20, 480, 340)
WRIST_BOX = (160, 380, 480, 700)
SOURCE_SIDE = 256  # render.py: "source frames are 256x256"

OPEN_X = (-0.0665, 0.0957)
OPEN_Y = (0.1661, 0.3395)

# Turn -> what the model asserted, and the measured EEF at that moment.
#
# CAVEAT on frame vs decision: the recorder writes frame N *after* turn N's
# tool ran (harness: call -> execute -> attach result+images -> record), so the
# decision at turn N was made on frame N-1.  Both are listed: 7/14/20/25/30 are
# the frames the model actually saw when it asserted "已到达篮子正上方", and
# 8/15/21/26/31 are the states right after it released.
TURNS = {
    7: "DECISION FRAME for t8's claim（此时爪上仍夹着罐子）— 实测 y 差 20.4cm, x 差 5.9cm",
    14: "DECISION FRAME for t15's claim（爪上仍夹着）— 实测 y 差 5.3cm, x 差 6.7cm",
    20: "DECISION FRAME for t21's claim（爪上仍夹着）— 实测 y 在范围内, x 差 7.1cm",
    25: "DECISION FRAME for t26's claim（爪上仍夹着）— 实测 y 在范围内, x 差 7.4cm",
    30: "DECISION FRAME for t31's claim（爪上仍夹着）— 实测 y 在范围内, x 差 7.4cm",
    8: "POST-ACTION: t8 open 之后（实测越出 x 5.9cm / y 20.5cm）",
    15: "POST-ACTION: t15 open 之后（x 6.7cm / y 5.1cm）",
    21: "POST-ACTION: t21 open 之后（x 7.0cm / y 0）",
    26: "POST-ACTION: t26 open 之后（x 7.3cm / y 0）",
    31: "POST-ACTION: t31 open 之后（x 7.2cm / y 0）",
    32: "CONTROL: 真的在篮口内（实测 x=+0.0132, y=+0.2532），模型却说“物体仍未落入篮中”",
    60: "CONTROL: 明显在远处角落（实测 x 差 48.7cm）",
}

QUESTION = (
    "以上两张图是同一时刻机器人的画面：第 1 张是全局第三人称视角，第 2 张是装在夹爪上的腕部相机、朝正下方看。\n"
    "只根据这两张画面回答，不要根据常识或任务目标猜测：\n"
    "1. 分别描述两张画面里各自能看到的主要内容。\n"
    "2. 第 2 张（腕部）画面正中央的下方是什么？\n"
    "3. 画面里那个容器（篮子）的开口，是否就在夹爪的正下方？如果不在，请说明它相对夹爪的方位，"
    "以及大概有多远（用画面里物体的尺寸作参照来估计）。\n"
    "4. 如果现在张开夹爪，手里的物体会落进那个容器里吗？\n"
    "请简短回答，逐条对应。"
)

NEUTRAL_SYSTEM = (
    "你是一个机器人视觉助手。你只能依据给出的画面来回答，画面里看不出来的事情就直接说看不到，不要臆测。"
)

# --- CoT experiment ---------------------------------------------------------
# v2's system prompt currently FORBIDS multi-part reasoning:
#     "输出格式：先用**一句话以内**说明判断，紧接着立即写出完整的工具调用。
#      不要写多段推理、不要重复已经说过的内容。"
# The single-turn probes showed the model CAN judge "容器不在正下方" when asked --
# but in the loop it never runs that check (0/80 turns doubted the release
# alignment) and instead emits a one-line preamble that contradicts the judgment.
# So the test is whether forcing a written 判定 section makes the call follow it.
V2_FORMAT_OLD = (
    "输出格式：先用**一句话以内**说明判断，紧接着立即写出完整的工具调用。\n"
    "不要写多段推理、不要重复已经说过的内容。"
)

V2_FORMAT_COT = (
    "输出格式：每次回复严格按下面三段写，三段都要有，标题照拄：\n"
    "现状：（这一轮画面里的事实：爪上有没有东西；夹爪正下方是什么；容器（篮子）相对爪子在哪个方向、大约多远）\n"
    "判定：（如果此刻张开夹爪，手里的物体会落到哪里？会不会落进容器？）\n"
    "动作：（下一步调用哪个工具、参数是多少；必须与上面的判定一致——判定说不会落进容器，就不要在这一步 open）\n"
    "三段写完后，紧接写出完整的工具调用。除这三段外不要写其他内容。"
)

# The `leading` wording above contains "是否就在夹爪的正下方？如果不在，请说明方位",
# which presupposes "not below" as a live answer -- and the t32 control shows the
# model will say "not below" even when it IS below, so that wording cannot by
# itself prove spontaneous detection.  This variant asks the same three facts with
# no presupposition, and question 3 is phrased symmetrically ("是否会进入").
QUESTION_OPEN = (
    "以上两张图是同一时刻机器人的画面：第 1 张是全局第三人称视角，第 2 张是装在夹爪上的腕部相机、朝正下方看。\n"
    "只根据这两张画面回答，不要根据常识或任务目标猜测：\n"
    "1. 夹爪正下方有什么？请具体描述你看到的东西。\n"
    "2. 画面里那个容器（篮子）在什么位置？相对夹爪是哪个方向、大概多远（用画面里物体的尺寸作参照）？\n"
    "3. 如果此刻张开夹爪，手里的物体会落到哪里？请给出具体位置，并说明它是否会进入那个容器。\n"
    "请简短回答，逐条对应。"
)


def frames_for(turns: list[int]) -> dict[int, tuple[Image.Image, Image.Image]]:
    import imageio.v3 as iio

    # Frame index == turn number: the recorder writes one frame per turn plus an
    # initial and a final one, so 80 turns -> 82 frames.
    needed = set(turns) | {0}
    got: dict[int, tuple[Image.Image, Image.Image]] = {}
    for i, arr in enumerate(iio.imiter(str(VIDEO))):
        if i in needed:
            img = Image.fromarray(np.asarray(arr))
            agent = img.crop(AGENT_BOX).resize((SOURCE_SIDE, SOURCE_SIDE), Image.LANCZOS)
            wrist = img.crop(WRIST_BOX).resize((SOURCE_SIDE, SOURCE_SIDE), Image.LANCZOS)
            got[i] = (agent, wrist)
        if i > max(needed):
            break
    missing = [t for t in turns if t not in got]
    if missing:
        raise SystemExit(f"video has no frames for turns {missing}")
    return got


def measured() -> dict[int, str]:
    import json

    d = json.load(open(EPISODE))
    out: dict[int, str] = {}
    for e in d["transcript"]:
        if "result" not in e:
            continue
        ee = ((e.get("structured_result") or {}).get("measurement") or {}).get("eef_end")
        if not ee:
            continue
        x, y = ee[0], ee[1]
        xo = max(OPEN_X[0] - x, x - OPEN_X[1], 0.0) * 100
        yo = max(OPEN_Y[0] - y, y - OPEN_Y[1], 0.0) * 100
        out[e["turn"]] = f"x={x:+.4f} y={y:+.4f}  越出篮口 x {xo:.1f}cm / y {yo:.1f}cm"
    return out


def loop_action(turn: int) -> str:
    """What the 80-turn run actually called at `turn` (for the A/B comparison)."""
    import json as _json
    import re as _re

    d = _json.load(open(EPISODE))
    for e in d["transcript"]:
        if e.get("turn") != turn or "assistant" not in e:
            continue
        m = _re.search(r"<function=(\w+)>", e["assistant"])
        params = dict(_re.findall(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", e["assistant"], _re.S))
        return f"{m.group(1) if m else '?'} {params}"
    return "-"


def turn_result_text(turn: int) -> str:
    """The user-visible tool result the model read when it decided the NEXT turn."""
    import json as _json

    d = _json.load(open(EPISODE))
    out = ""
    for e in d["transcript"]:
        if e.get("turn") == turn and "result" in e:
            out = e["result"]
    return out


_GRIP_OPEN, _GRIP_EMPTY = 0.034, 0.006
_WIDTH_RE = None


def _width(turn: int, keep: dict | None = None) -> float | None:
    import json as _json
    import re as _re

    txt = turn_result_text(turn)
    m = _re.search(r"finger width ([\d.]+)", txt)
    return float(m.group(1)) if m else None


def width_timeline() -> dict[int, float | None]:
    """Finger width per turn, carrying the last gripper reading forward.

    Only gripper actions print "finger width"; the live `format_state` line prints
    it every turn from the current qpos, so the value has to be carried forward or
    move/approach turns read as 未知.
    """
    import json as _json

    d = _json.load(open(EPISODE))
    last: float | None = None
    out: dict[int, float | None] = {}
    for e in sorted((x for x in d["transcript"] if "turn" in x), key=lambda x: x["turn"]):
        w = _width(e["turn"])
        if w is not None:
            last = w
        out[e["turn"]] = last
    return out


def grasp_timeline() -> dict[int, bool]:
    """Reproduce harness `_grasped`: set by a `close` that bit something, cleared by `open`."""
    import json as _json

    d = _json.load(open(EPISODE))
    state = False
    out: dict[int, bool] = {}
    entries = sorted((e for e in d["transcript"] if "turn" in e), key=lambda e: e["turn"])
    for e in entries:
        # `verify` turns carry no structured_result, so the state has to be
        # carried forward across them rather than dropped.
        sr = e.get("structured_result") or {}
        action = (sr.get("requested") or {}).get("action")
        w = _width(e["turn"])
        if sr.get("tool") == "gripper" and action == "close":
            state = bool(w is not None and _GRIP_EMPTY < w < _GRIP_OPEN)
        elif sr.get("tool") == "gripper" and action == "open":
            state = False
        out[e["turn"]] = state
    return out


def reconstructed_message(turn: int, grasped: bool) -> str:
    """Rebuild harness lines 408-415 verbatim: result text + json + format_state.

    The episode transcript stores only `result` (the human sentence), so the other
    two parts have to be rebuilt from the measurements; without this the probe
    feeds a message that never says 爪上：<物体>, and the model reasonably concludes
    "手里没有东西" -- which is an artefact of the probe, not of the model.
    """
    import json as _json

    d = _json.load(open(EPISODE))
    result_json = "{}"
    eef = None
    entry = None
    for e in d["transcript"]:
        if e.get("turn") == turn and "result" in e:
            entry = e
    if entry is not None:
        sr = entry.get("structured_result") or {}
        result_json = _json.dumps(sr, ensure_ascii=False, separators=(",", ":"))
        eef = (sr.get("measurement") or {}).get("eef_end") or (sr.get("measurement") or {}).get("eef_start")

    w = width_timeline().get(turn)
    if w is None:
        grip = "未知"
    elif w >= _GRIP_OPEN:
        grip = "张开"
    elif w <= _GRIP_EMPTY:
        grip = "闭合（空）"
    else:
        grip = "半闭（指间有物体）"
    # The real line also carries the payload's distance from the gripper centre;
    # that needs the live object pose, which the transcript does not store, so it
    # is omitted (the load-bearing part is 爪上：<name> vs 爪上：空).
    hold = "爪上：alphabet_soup" if grasped else "爪上：空（没有物体跟着手）"
    where = f"({eef[0]:.4f}, {eef[1]:.4f}, {eef[2]:.4f})" if eef else "(未知)"
    return f"{turn_result_text(turn)}\n{result_json}\n机械爪位置 {where} m；夹爪：{grip}；{hold}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["neutral", "v2", "both"], default="both")
    # 420 truncated several answers mid-sentence, so "did it then call open?"
    # could not be read off the output.  Long Chinese answers need ~700.
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--turns", default="", help="comma-separated override of TURNS keys")
    # leading = the wording with "如果不在" (presupposes not-below);
    # open    = same three facts asked without presupposition.
    ap.add_argument("--question", choices=["leading", "open"], default="leading")
    # input=question : images + an explicit question (labelled perception probe)
    # input=result   : images + that turn's REAL tool-result text, no question at
    #                  all -- this is what the loop actually feeds the model, so
    #                  the only variable left is the output-format rule.
    ap.add_argument("--input", choices=["question", "result"], default="question")
    # Faithfulness: rebuild the full harness message (result + json + format_state)
    # instead of the bare result sentence the transcript stores.
    ap.add_argument("--no-state", action="store_true", help="omit the rebuilt format_state line")
    ap.add_argument("--format", choices=["v2", "cot", "both"], default="v2")
    ap.add_argument("--load", default=os.environ.get("LOAD", "bf16"))
    ap.add_argument("--gpu-mem-gib", type=float, default=float(os.environ.get("GPU_MEM_GIB", 29)))
    args = ap.parse_args()

    if args.turns:
        keep = [int(x) for x in args.turns.split(",") if x.strip()]
        TURNS = {k: f"(specified) frame {k}" for k in keep}  # noqa: F841 - local override
        globals()["TURNS"] = TURNS

    model_dir = os.environ.get("ARM_AGENT_MODEL_DIR", str(REPO.parent / "qwen35_demo" / "models" / "Qwen3.5-9B"))
    truth = measured()
    frames = frames_for(sorted(TURNS))

    systems = {}
    if args.condition in ("neutral", "both"):
        systems["neutral"] = NEUTRAL_SYSTEM
    if args.condition in ("v2", "both"):
        from arm_agent.agent.prompts import get_system_prompt

        base = get_system_prompt("v2")
        if V2_FORMAT_OLD not in base:
            raise SystemExit("v2 format paragraph changed; update V2_FORMAT_OLD")
        formats = ["v2", "cot"] if args.format == "both" else [args.format]
        for fmt in formats:
            systems[f"v2/{fmt}"] = base.replace(V2_FORMAT_OLD, V2_FORMAT_COT) if fmt == "cot" else base

    model = QwenAgentModel(
        model_dir=model_dir,
        load=args.load,
        gpu_mem_gib=args.gpu_mem_gib,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        image_max_side=160,  # same as the run (harness default)
    )
    model.load_model()

    if args.input == "result":
        timeline = grasp_timeline()
    for name, system in systems.items():
        for turn in sorted(TURNS):
            agent, wrist = frames[turn]
            if args.input == "result":
                user_text = (
                    turn_result_text(turn) or "（无工具结果）"
                    if args.no_state
                    else reconstructed_message(turn, timeline.get(turn, False))
                )
            else:
                user_text = QUESTION_OPEN if args.question == "open" else QUESTION
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": agent},
                        {"type": "image", "image": wrist},
                        {"type": "text", "text": user_text},
                    ],
                },
            ]
            reply = model.chat(messages, None)
            print("=" * 100)
            print(f"[{name}/{args.question}/{args.input}/{args.format}] turn {turn}  {TURNS[turn]}")
            print(f"  实测: {truth.get(turn, '-')}")
            print(f"  循环里 t{turn + 1} 实际动作: {loop_action(turn + 1)}")
            if args.input == "result" and not args.no_state:
                print(f"  喂入的状态行: …{user_text.splitlines()[-1]}")
            print("-" * 100)
            print(reply.strip())
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
