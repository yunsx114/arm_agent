"""The 8-tool surface exposed to the model, plus the system prompt.

Design decisions live in DESIGN.md 2.4: semantic primitives (direction + amount
in the world frame), never raw OSC deltas; <=10 tools; every motion tool result
feeds MEASURED numbers back so the model can self-correct (E5 showed its
direction semantics cannot be trusted).

The tool schemas are passed to tokenizer.apply_chat_template(tools=...) which
renders the "# Tools" block; descriptions are Chinese because the whole
conversation is Chinese and the template output is bilingual-tolerant.
"""

from __future__ import annotations

from typing import Any
from arm_agent.agent.tool_protocol import CANONICAL_TOOLS

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": (
                "重新获取当前画面（机械臂不会移动）。camera=agent 为全局第三视角，"
                "camera=wrist 为机械爪上的近景相机（对准物体时更有用）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera": {"type": "string", "enum": ["agent", "wrist"], "description": "相机视角"},
                },
                "required": ["camera"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate",
            "description": (
                "用物体检测器报告指定物体当前的世界坐标（x,y,z，单位米）。"
                "常用名：alphabet_soup（要抓的罐头）、basket（篮子）、milk、tomato_sauce 等。"
                "抓取前先用它确认物体的准确位置，不要仅凭画面猜测。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "物体名（可用英文关键词）"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "descend_to",
            "description": (
                "自动对准并下降到指定物体（全程闭环纠偏，比手动 step-by-step 下降可靠得多）。"
                "z_offset_cm 是最终停在物体中心上方多少厘米：抓取时用 1.2；放入宽容器时可用更大值。"
                "抓取的标准做法：descend_to(alphabet_soup, 1.2) → set_gripper close。"
                "**注意**：不要用负 z_offset 硬压（会从侧面刮到物体、把它水平推开）；夹空后先 move +z 抬起再 align_xy。"
                "也不要下降到紧贴篮子的物体上（距篮子中心 6cm 内下降会把篮子推倒）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "物体名，例如 alphabet_soup / basket"},
                    "z_offset_cm": {"type": "number", "description": "停在物体中心上方的高度，默认 1.2"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "align_xy",
            "description": (
                "把机械爪自动、精准地对准到指定物体的正上方（服务端闭环微调，精度约 2mm）。"
                "抓取前先用它定位对准，再 move -z 下降。不要手动用小距离 move 反复对——那是它的工作。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "物体名，例如 alphabet_soup"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "读取机械爪末端当前状态：位置坐标(x,y,z，单位米)、夹爪开合。不移动机械臂。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": (
                "让机械爪沿世界坐标轴平移指定距离（0.1~15 厘米）。"
                "direction:+x/-x/+y/-y/+z/-z；+z 向上，-z 向下。"
                "尽量一次走完需要的距离（例如要移动 13 厘米就直接 distance_cm=13），"
                "不要拆成很多次小移动——每一步都有小概率把物体震掉。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["+x", "-x", "+y", "-y", "+z", "-z"]},
                    "distance_cm": {"type": "number", "description": "平移距离，单位厘米，范围 0.1~15"},
                },
                "required": ["direction", "distance_cm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate",
            "description": (
                "让机械爪绕世界坐标轴旋转。axis=yaw 绕竖直轴（水平转动），"
                "pitch 绕 y 轴（前后翻转），roll 绕 x 轴（左右侧倾）。angle_deg 范围 -90~90。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "axis": {"type": "string", "enum": ["yaw", "pitch", "roll"]},
                    "angle_deg": {"type": "number", "description": "旋转角度，单位度，范围 -90~90"},
                },
                "required": ["axis", "angle_deg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_gripper",
            "description": "张开或闭合夹爪。抓取物体前先 open，对准物体后 close，提起前确保已闭合。",
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
            "description": "声明任务已完成。系统会自动检查是否真的成功，如果未成功会告诉你继续。",
            "parameters": {
                "type": "object",
                "properties": {"answer_note": {"type": "string", "description": "一句话说明你做了什么"}},
                "required": ["answer_note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_skill",
            "description": (
                "把这次任务里学到的可复用经验写进技能库，供以后的任务参考。"
                "只有确实有用的经验才写（如抓取技巧、坐标系规律）。name 用短横线小写英文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "技能名，小写字母和短横线"},
                    "content": {"type": "string", "description": "经验正文（简短、具体、可操作）"},
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "读取技能库里某个技能的完整内容。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]

LEGACY_TOOL_SCHEMAS = TOOL_SCHEMAS
# v2 surface. The descriptions now state the SAME limits the validator enforces
# (`tool_protocol.REQUIRED` / `ENUMS`, the 0.1-15 cm and +-90 deg ranges). They
# used to be hidden: the model could only discover them by being rejected, and
# every rejection costs a whole turn. `height_offset_cm` was also unitless and
# had no documented default -- which is how a run spent 26 turns parking 5 cm
# above the object and closing on air (`gripper` width 0.0007, 26/26 calls).
TOOL_SCHEMAS_V2: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "observe",
        "description": (
            "读取当前状态和场景证据，不移动机械臂。给 target 时返回该物体的坐标与偏差；"
            "给 camera 时重新渲染并返回该视角的新画面（不给 camera 就不会产生新画面）。"
        ),
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "物体名，例如 alphabet soup / basket"},
            "camera": {"type": "string", "enum": ["agent", "wrist"], "description": "要刷新的相机视角"},
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "approach",
        "description": (
            "对准目标并下降到它中心上方 height_offset_cm 厘米处（默认 1.2；正值=停在物体上方，"
            "抓取时用 1.2 左右；不要用负值硬压）。不操作夹爪。"
        ),
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "物体名"},
            "height_offset_cm": {"type": "number", "description": "最终停在物体中心上方多少厘米，默认 1.2"},
        }, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "move",
        "description": "沿世界坐标轴平移，距离 0.1~15 厘米（+z 向上，-z 向下）。返回实测位移，可能少于请求值。",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["+x", "-x", "+y", "-y", "+z", "-z"]},
            "distance_cm": {"type": "number", "description": "平移距离，单位厘米，范围 0.1~15"},
        }, "required": ["direction", "distance_cm"]}}},
    {"type": "function", "function": {
        "name": "rotate",
        "description": "绕世界坐标轴旋转末端，角度范围 -90~90 度。返回实测结果。",
        "parameters": {"type": "object", "properties": {
            "axis": {"type": "string", "enum": ["yaw", "pitch", "roll"]},
            "angle_deg": {"type": "number", "description": "旋转角度，单位度，范围 -90~90"},
        }, "required": ["axis", "angle_deg"]}}},
    {"type": "function", "function": {
        "name": "gripper",
        "description": "打开或闭合夹爪，并返回实测指宽与是否夹住物体（两指完全闭合=空夹）。",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["open", "close"]},
        }, "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "verify",
        "description": "请求环境验证任务是否完成。成功与否只由环境真值决定，claim 只是标签，不影响判定。",
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string", "enum": ["placed", "grasped", "task_complete"]},
        }, "required": []}}},
]

SYSTEM_PROMPT = """你是机械臂智能体的决策核心。你通过调用工具观察场景、移动机械臂，完成"把指定物体拿起来放进篮子"的任务。

## 工作方式
- 每次回复只调用**恰好一个**工具。可以在工具调用前用一两句话说明判断，之后不要再加解释。
- 工具执行后你会收到：结果说明 + 机械爪的最新位置坐标 + 最新画面。请用这些**真实数字**计划下一步。
- 判断方位时不要只凭图像感觉：结合画面和 get_state 给出的坐标做决定，不确定就多移动几次小步试探。

## 坐标系（世界系，单位米）
- z 轴向上：桌面大约 z≈0，+z 向上，-z 向下。
- 相机在场景一侧俯视：x 越大越靠近相机（画面越靠下），y 越大越靠画面右侧。
- 机械爪初始悬停在桌面上方 z≈0.25，物体都躺在桌面上（z≈0.03~0.10）。
- **先 locate 拿到物体的确切坐标，再算出差值逐步靠近**，不要凭画面猜深度。

## 典型抓取流程
1. locate：拿到目标物体的坐标与抓取高度提示。
2. **descend_to(物体, 1.2)**：一步完成“对准 + 下降到抓取高度”（全程自动纠偏）。不要自己拼多个小 move 去对准，那个又慢又不准。
3. set_gripper close 抓取。
   - **看结果判断是否夹到**：提示里会直接告诉你“✓ 夹住了”或“⚠ 夹空了”。
   - **夹空后重试**：先 `move +z 5` 抬起 → `align_xy` → `descend_to(name, 1.2)`。
     **不要用负 z_offset 硬压**——实测会把物体横向推开（v6 推走 5.4cm，最后撞倒篮子）。
4. **提高：把物体底部抬到篮口以上（关键，抬不够横移就会被刮掉）**。
   - 实测**篮口在 z≈0.115**。罐头高约 10.7cm（半高 0.054），所以**物体中心必须抬到 z ≥ 0.17**，
     即从抓取高度（z≈0.05）起算要 **move +z 至少 12~15cm**。常见错误：只抬 10cm 就去横移——
     看起来「已经很高了」，但底部仍在篮口之下，从篮子上方扫过时一定会被刮掉。
   - **判据**：横移前先 `locate 物体`，看它自己的 z；**z < 0.17 就继续 move +z**。
   - **不要超过 z≈0.30**：再高会接近机械臂工作空间上限，之后所有 move 都可能失效。
5. locate basket → **descend_to(basket, 15)** 降到篮子口上方（**不要用 10 或更小**：实测 z_offset<12cm 时爪子会压下篮口、把篮子推开，篮子一旦移位就再也放不进去）。
6. **set_gripper open 松开物体**。
7. **move +z 10 抬起**，然后**立刻 declare_done**（系统会验证是否真的放好了）。
   - **open 之后结果里会显示“爪上：空”，这是正常的**（你是主动松手把物体放进篮子的）。
     **不要再去抓它**，直接 declare_done；若要确认位置用 locate 看一眼即可。
   - 只有在你**没有主动 open** 的情况下看到“爪上：空”，才是运输途中掉了；那时才 locate 找回重抓。

## 经验提示
- **单次 move 可以走 0.1~15 厘米，尽量一次走完**：拆成很多步小移动时，每一步都有小概率把物体震掉，而移动次数直接决定“掉了没”的概率。
- **偏差小于 5cm 时按实际偏差大小移动**（例如 dx=+1.2cm 就 move +x 1.2），不要固定用 5cm——否则会冲过目标、来回震荡。
- locate 的结果里直接给出了偏差和建议动作，优先照做。
- 夹爪闭合后会有一小段滑移，抓取后可以稍等/微调再提起。
- **保护篮子**：篮子会被推倒。实测**距篮子中心 4cm 以内下降必把它推开 10~37mm**（6cm 外基本不碰）。
  不要在篮子附近下降或横移。
- **够不到时**（结果里出现“够不到”/“只走了…”，或 dz 差好几厘米降不下去）：该位置超出工作空间，
  **再试也不会更低**。用爪子在物体**侧面**轻推（`move ±x` 5~10cm）挪到别处再抓。
- 如果动作结果与预期不符（比如移动方向反了），立刻修正，不要重复同一个失败动作。
- **每次动作后看结果末尾的“爪上：…”**：
  * 如果你**刚执行完 open**（主动放开），显示“爪上：空”是**正常的**，接下来应该 **declare_done 验证**，不要去抓物体。
  * 如果你**没有 open** 却是“爪上：空”，那才是运输中掉了：**不要惊慌、不要反复 move**，直接 locate 找到物体当前位置，再 descend_to + close 重新夹起。
- **完成任务后一定要 declare_done**：系统才会判定成功；不调用就永远算未完成。
- 你可以用 write_skill 记录经验，之后的任务里 read_skill 读取。
"""

# ---------------------------------------------------------------------------
# PLAIN prompt -- P0a of the harness-stripping plan.
#
# Rationale: the full prompt above accumulated ~1400 tokens of
# SCENE-SPECIFIC PATCHES, each reverse-engineered from a single failed episode:
# basket rim at z=0.115, can half-height 0.054, "centre z >= 0.17", "within 4 cm
# of the basket centre", "z_offset < 12 cm", a 7-step recipe, plus citations of
# past crashes. Those patches (a) overfit 5-10 observations, (b) have already
# contradicted each other twice, and (c) do the model's planning for it -- so the
# runs stopped measuring the model at all.
#
# This version keeps only what the model CANNOT discover on its own:
#   - the task,
#   - the world-frame convention and the camera mapping,
#   - what each tool does.
# Everything about HOW to do it (order of operations, heights, distances, danger
# radii) is deliberately absent -- that is the thing under test.
#
# Selected with `--prompt-mode plain`.
SYSTEM_PROMPT_PLAIN = """你是机械臂智能体的决策核心。任务是：把指定物体拿起来，放进篮子。

## 工作方式
- 每次回复只调用**恰好一个**工具。可以在工具调用前用一两句话说明判断，之后不要再加解释。
- 工具执行后会返回：结果说明 + 机械爪的最新位置坐标 + 最新画面。用这些**真实数字**计划下一步。
- 不确定就多观察、多小步试探；每步之后根据最新数字重新判断。

## 坐标系（世界系，单位米）
- z 轴向上：桌面大约 z≈0，+z 向上，-z 向下。
- 相机在场景一侧俯视：x 越大越靠近相机（画面越靠下），y 越大越靠画面右侧。
- 机械爪初始悬停在桌面上方 z≈0.25。

## 工具
- `look` / `get_state`：观察，不移动机械臂。
- `locate`：报告某个物体当前的世界坐标。场景里有哪些物体，可先用它确认。
- `move` / `rotate`：沿世界坐标轴平移 / 旋转末端（单次上限见参数说明）。
- `descend_to` / `align_xy`：自动对准并下降到指定物体（服务端闭环，比手动小步移动准）。
  `z_offset_cm` 是**最终停在物体中心上方多少厘米**：贴着它抓取时用小值，只想悬停在它上方时用大值。
- `set_gripper`：`close` 夹住、`open` 松开。**夹取要把动作连起来**：先对准并降到物体上
  （`descend_to`），再 `close`；结果会告诉你是否真的夹住了（夹空要重新对准再试）。
- `declare_done`：认为完成时调用；系统会用真值验证并告诉你结果。

## 任务结构
1. 找到目标物体，把它**抓起来**。
2. 把它**搬到篮子上方**。
3. **松开夹爪**，让它落进篮子。

按顺序做，每步完成后再进入下一步。
**已经抓起来之后不要再去重复抓取动作**——对同一个物体再 `descend_to` / `close` 会把它碰掉。
结果里的「爪上：…」会告诉你手里有没有东西；手里有东西时就应该直接去做第 2 步。

完成后调用 `declare_done`。
"""

SYSTEM_PROMPT_V2 = """你是机械臂任务决策核心。目标是完成当前任务。

每次只调用一个工具。工具结果是环境测量，不是行动建议；不要假设动作一定完成。
如果动作被拒绝或只完成一部分，根据返回的状态和事件自行选择下一步。
不要发明工具、参数或并行调用。verify 的成功与否只由环境验证。

输出格式：先用**一句话以内**说明判断，紧接着立即写出完整的工具调用。
不要写多段推理、不要重复已经说过的内容。

任务结构（每步完成后再进下一步，不要跳步也不要重复已完成的一步）：
找到目标 → 张开夹爪并对准目标 → 合上夹爪夹住 → 抬高 → 移到篮子上方 → 张开夹爪放下。

事实（都是测量的，不是建议）：
- 夹爪必须**先张开**才夹得住东西。结果里写「两指完全闭合」时，再 close 等于空夹；
  这种状态下下降只会把物体推开。
- 已经夹着东西时，不要再往桌面上下降，否则会把夹着的东西刮掉；要先横向移到目标上方。
- 指宽读数：≈0.0005 = 两指完全闭合（空夹）；0.02~0.03 = 指间有物体（夹住了）；
  ≈0.036 = 张开。每次工具结果里都给出当前指宽，用它判断手里有没有东西。
- **你主动 open 之后「爪上：空」是预期的**，不代表物体掉了。只有在你没有 open 的
  情况下看到「爪上：空」，才是运输中掉了。
- 结果里写「未到达 / 卡住 / 只走了 X cm」都表示那个动作没有达成；以实测数字为准。

坐标系：世界坐标中 +z 向上，位置单位米，move 距离单位厘米。
需要新画面时必须用 observe 的 camera 参数，否则画面不会更新。
工具职责：
- observe：读取状态和场景证据，不移动。
- approach：到目标上方并下降，不操作夹爪。
- move：执行明确的相对平移。
- rotate：执行明确的姿态旋转。
- gripper：打开或闭合并报告测量到的夹爪状态。
- verify：让环境检查任务是否完成。
"""


PROMPT_MODES = {
    "full": SYSTEM_PROMPT,
    "plain": SYSTEM_PROMPT_PLAIN,
    "v2": SYSTEM_PROMPT_V2,
}


def get_system_prompt(mode: str = "full") -> str:
    """Pick the system prompt. Unknown modes fall back to `full`."""
    return PROMPT_MODES.get(str(mode).strip().lower(), SYSTEM_PROMPT)


def get_tool_schemas(protocol: str = "v2") -> list[dict[str, Any]]:
    return TOOL_SCHEMAS_V2 if str(protocol).strip().lower() == "v2" else LEGACY_TOOL_SCHEMAS
