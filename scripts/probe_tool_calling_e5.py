"""Phase 0 · E5 probe: real Qwen3.5-4bit tool-calling generation (runs on GPU).

Feeds the E1 scene image + task + state to the model with the 8-tool schema and
walks a scripted multi-turn conversation (simulated tool results), checking:
  1. does the model emit well-formed <tool_call><function=..> blocks?
  2. are the tool choices and arguments sensible for the goal?
  3. does the protocol survive across turns?

Run on gpu026 (see scripts/run_e5_on_gpu.sh). Loads via qwen35_demo/chat.py's
load_model so the 4bit recipe has a single source of truth.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QWEN_DEMO = REPO.parent / "qwen35_demo"
sys.path.insert(0, str(QWEN_DEMO))

MODEL_DIR = QWEN_DEMO / "models" / "Qwen3.5-9B"
AGENTVIEW = REPO / "outputs" / "smoke" / "e1_libero_object_0_agentview.png"
SCENE_JSON = REPO / "outputs" / "smoke" / "e1_scene_state.json"

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([\w.-]+)>")
PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Parse Qwen3.5 XML tool calls: <tool_call><function=f><parameter=k>v</parameter></function></tool_call>."""
    calls = []
    for block in TOOL_CALL_RE.findall(text):
        fn = FUNCTION_RE.search(block)
        if not fn:
            continue
        args = {m.group(1): m.group(2) for m in PARAM_RE.finditer(block)}
        calls.append({"name": fn.group(1), "args": args})
    return calls


def truncate_to_first_call(text: str) -> str:
    """Keep the natural-language prefix + first complete tool_call.

    The 4bit model does not reliably stop at the end of its assistant turn
    (its generation_config lacks <|im_end|> as eos; see chat.py which sets
    eos_token_id explicitly), so it can keep hallucinating the *next* turns.
    The harness must trim to the first call before echoing anything back.
    """
    end = text.find("</tool_call>")
    if end == -1:
        return text
    return text[: end + len("</tool_call>")]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "获取最新画面。camera=agent 为全局第三视角，camera=wrist 为机械爪上的相机。",
            "parameters": {
                "type": "object",
                "properties": {
                    "camera": {"type": "string", "enum": ["agent", "wrist"]},
                },
                "required": ["camera"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "读取机械爪末端的结构化状态：位置(x,y,z,米)、朝向、夹爪开度、最近一次工具执行结果。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "让机械爪沿世界坐标轴平移。direction 是方向，distance_cm 是距离(0.5~5厘米)。"
            "例：move +x 3 = 向 x 正方向 3 厘米。",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["+x", "-x", "+y", "-y", "+z", "-z"]},
                    "distance_cm": {"type": "number"},
                },
                "required": ["direction", "distance_cm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate",
            "description": "绕世界坐标轴旋转机械爪。axis=yaw/pitch/roll，angle_deg 范围 -30~30。",
            "parameters": {
                "type": "object",
                "properties": {
                    "axis": {"type": "string", "enum": ["yaw", "pitch", "roll"]},
                    "angle_deg": {"type": "number"},
                },
                "required": ["axis", "angle_deg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_gripper",
            "description": "张开或闭合夹爪。",
            "parameters": {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": ["open", "close"]}},
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_done",
            "description": "声明任务已完成。系统会检查是否真的成功。",
            "parameters": {
                "type": "object",
                "properties": {"answer_note": {"type": "string"}},
                "required": ["answer_note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_skill",
            "description": "把本次任务中学到的可复用经验写入技能库（name 短横线小写，content 为经验正文）。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "content": {"type": "string"}},
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "读取技能库中某个技能的完整内容。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是一个机械臂智能体的决策核心。你可以调用工具来观察场景和操作机械臂，完成给定的抓取放置任务。

工作方式：
- 每次回复必须调用恰好一个工具（不要输出多个 tool_call）。
- 调用工具前，允许用一两句话说明你的判断，然后给出工具调用。不要在工具调用后再加解释。
- 坐标系：世界系 z 轴向上，桌面在 z≈0；x 正方向朝画面右侧，y 正方向朝画面里侧（远离相机）。
- get_state 会给出机械爪当前位置；你的目标是先移动到目标物体上方，再下降、闭爪、提起，最后移动到篮子上方放下。
- 动作幅度限制：单次 move 最多 5 厘米——远距离请多走几步。
- 当物体已放入篮子，调用 declare_done。
"""


def fmt_state(state: dict) -> str:
    eef = state["eef_pos"]
    return (
        f"机械爪位置: ({eef[0]:.3f}, {eef[1]:.3f}, {eef[2]:.3f}) 米\n"
        f"夹爪开度: {'张开' if abs(state['gripper_qpos'][0]) > 0.015 else '闭合'}"
    )


def main() -> int:
    import chat  # qwen35_demo/chat.py

    t0 = time.time()
    tokenizer, model, processor = chat.load_model(str(MODEL_DIR), load="4bit", gpu_mem_gib=9.5)
    print(f"[e5] model loaded in {time.time() - t0:.0f}s", flush=True)

    from PIL import Image

    image = Image.open(AGENTVIEW).convert("RGB")
    state = json.loads(SCENE_JSON.read_text())

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"任务: {state['language']}\n"
                        f"[上图 = agent 视角最新画面]\n"
                        f"{fmt_state(state)}\n"
                        f"请决定下一步动作。"
                    ),
                },
            ],
        },
    ]

    # Scripted tool results for the walk: each entry replaces what a real tool
    # would return so the protocol (state echo) can be exercised multi-turn.
    canned_results = [
        "[move] 完成。机械爪位置: (-0.119, -0.180, 0.290) 米, 夹爪: 张开",
        "[get_state] 机械爪位置: (-0.118, -0.240, 0.290) 米, 夹爪: 张开",
    ]

    all_ok = True
    for turn in range(3):
        t0 = time.time()
        text = tokenizer.apply_chat_template(
            conversation, tools=TOOLS, tokenize=False, enable_thinking=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.05,
            # Both are required on this model+4bit stack:
            #  - eos_token_id: the checkpoint's generation_config uses 248044,
            #    but the real turn terminator is <|im_end|>=248046 (as chat.py
            #    already does). Without it generation runs past the turn.
            #  - stop_strings: belt-and-braces so a rogue continuation can't
            #    waste 200 tokens before we trim.
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["</tool_call>"],
            tokenizer=tokenizer,
        )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        reply = truncate_to_first_call(reply)
        dt = time.time() - t0

        calls = parse_tool_calls(reply)
        ok = len(calls) == 1
        all_ok = all_ok and ok
        print(f"\n===== turn {turn + 1} ({dt:.0f}s) =====", flush=True)
        print("--- raw reply ---")
        print(reply)
        print(f"--- parsed: {len(calls)} call(s) ---")
        for c in calls:
            print("   ", c["name"], c["args"])
        if not ok:
            print("   !! PROTOCOL FAILURE", flush=True)

        conversation.append({"role": "assistant", "content": reply})
        result = canned_results[turn] if turn < len(canned_results) else "[move] 完成。"
        # Only the first user turn carries the image: the processor gets exactly
        # one PIL image for exactly one vision placeholder. Later turns are text
        # (the image stays in context on the model side).
        conversation.append(
            {
                "role": "user",
                "content": f"工具结果: {result}\n请决定下一步动作。",
            }
        )

    print(f"\n[e5] protocol: {'ALL TURNS OK' if all_ok else 'HAD FAILURES'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
