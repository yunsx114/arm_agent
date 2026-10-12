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
from arm_agent.agent.tool_protocol import (
    CANONICAL_TOOLS,
    MODE_TOOLS,
    canonical_name,
    tools_for_mode,
)

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

# MOVE LADDER (档 D) -- the SAME `move` tool with a NAMED magnitude.
#
# WHY THIS SHAPE: measured, the model cannot turn what it sees into a world distance.
# In `pixels` mode its absolute reading carried a 30-50 px systematic bias (10-13 cm);
# in `raw` mode it put the can at y=-0.10 when the truth is y=-0.24. Both failures are
# the same missing ability. So the magnitude is moved OUT of the model's hands: it
# picks a direction (which it does well -- it names +y correctly) and a RUNG, and code
# owns the number. This is Jev's rule in `embodied_jev/hierarchical.py::assemble_motion`
# -- scale the magnitude a model chose, never replace the direction it chose.
#
# The three rungs also hand the model a closed loop it can run with no arithmetic at
# all: overshot -> downshift, too little -> upshift. The literals are accepted as
# aliases because a wrong word costs a turn and buys nothing.
MOVE_LADDER_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "move",
        "description": (
            "沿世界坐标轴平移。**你不报厘米，只报两件事**：哪个轴、画面里还差多少像素。"
            "direction:+x/-x/+y/-y/+z/-z；+z 向上，-z 向下。**这是唯一能改变高度的工具**。"
            "gap_px：你在全局画面里量到的、沿这个方向还差多少像素（看目标与青色十字的距离）。"
            "距离由代码按相机尺度换算并取整到三档（far=10cm / near=1cm / fine=1mm），不用你算。"
            "也可以直接指定档位：size=far/near/fine（不需要 gap_px）。"
            "返回里会告诉你实测位移、以及这一步在画面里实际移动了几像素，用它可以校准你的目测。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["+x", "-x", "+y", "-y", "+z", "-z"]},
                "gap_px": {
                    "type": "number",
                    "description": "画面里沿该方向还差多少像素（0.5~300）。例如目标在十字下方 80 像素就填 80",
                },
                "size": {
                    "type": "string",
                    "enum": ["far", "near", "fine"],
                    "description": "直接指定档位（可选，与 gap_px 二选一）：far=10厘米；near=1厘米；fine=0.1厘米(1毫米)",
                },
            },
            "required": ["direction"],
        },
    },
}

# The instruction that makes the ladder usable. It has to say the UPSHIFT/DOWNSHIFT rule
# explicitly, because "pick a rung" is meaningless without one: a model that always
# picks `near` will simply never arrive, and one that always picks `far` will orbit the
# target. The rule is stated as an error-budget decision so it stays a policy, not a
# number to compute.
MOVE_LADDER_APPENDIX = """

## 距离怎么报（本模式：你报像素，代码换算）

`move` 只问你两个东西：

1. `direction`：沿哪个世界轴（你从画面判断）。
2. `gap_px`：**你在全局画面里量到的、沿那个方向还差多少像素**（看目标与青色十字）。

**不要自己换算成厘米。**（实测：模型把「75px×3.8mm」算成了 2.8cm，真值是 28cm——一个数量级。
厘米数交给代码，它知道相机尺度。）

代码会把这个像素差换算成厘米，再取到三档里最近的一档：

| 你报的像素差 | 代码实际走 |
| --- | --- |
| ≥ 32 像素（约 12cm 以上） | `far`（10 厘米） |
| 3 ~ 32 像素（约 1~12cm） | `near`（1 厘米） |
| < 3 像素（小于 1cm） | `fine`（1 毫米） |

所以要做的只是**诚实报你量到的像素差**：近了它自然报得小、档位自然变小，
你不需要自己去“选档位”——那是代码的事。返回里会告诉你这一步实际移动了多少像素，
拿它校正你的目测：如果上一步走完目标跑到十字另一边了，说明你报大了；
如果几乎没动，说明报小了。

（也可以直接写 `size=far|near|fine` 跳过像素估，但一般不需要。）
"""


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
            "观察，不移动机械臂。三种用法，返回形式不同："
            "给 target → 返回该物体坐标与「偏差（物体 - 机械爪）」；"
            "给 camera → 重新渲染并返回该视角的新画面；都不给 → 返回当前机械爪状态。"
            "同时给 target 和 camera 时只按 target 处理。"
        ),
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "物体名，例如 alphabet soup / basket"},
            "camera": {"type": "string", "enum": ["agent", "wrist"], "description": "要刷新的相机视角"},
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "approach",
        "description": (
            "把夹爪**移到**目标物体处，停在它中心上方 height_offset_cm 厘米（默认 1.2）。"
            "它只做「对准 + 下降」这一件事：**不会抬起物体、不改变夹爪开合、不改变末端朝向**；"
            "已经在该高度时它不会产生任何位移。要抬高物体用 move +z。不要传负值硬压。"
        ),
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "物体名，把夹爪移到它上方"},
            "height_offset_cm": {"type": "number", "description": "停在物体中心（不是顶部）上方多少厘米，默认 1.2"},
        }, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "move",
        "description": (
            "沿世界坐标轴平移 0.1~15 厘米。**这是唯一能改变高度的工具**：+z 向上抬高、-z 向下。"
            "返回实测位移，可能少于请求值；不够时可以再走一次或分段。"
        ),
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["+x", "-x", "+y", "-y", "+z", "-z"]},
            "distance_cm": {"type": "number", "description": "平移距离，单位厘米，范围 0.1~15"},
        }, "required": ["direction", "distance_cm"]}}},
    {"type": "function", "function": {
        "name": "rotate",
        "description": "绕世界坐标轴旋转末端，角度范围 -90~90 度。只改变末端朝向，不改变位置。返回实测结果。",
        "parameters": {"type": "object", "properties": {
            "axis": {"type": "string", "enum": ["yaw", "pitch", "roll"]},
            "angle_deg": {"type": "number", "description": "旋转角度，单位度，范围 -90~90"},
        }, "required": ["axis", "angle_deg"]}}},
    {"type": "function", "function": {
        "name": "gripper",
        "description": (
            "开/关夹爪（raw 模式使用；truth/pixels 模式由 grasp/release 取代）。"
            "close = 合上直到碰到东西；open = 张开（空手时就是复位）。**只回报概念**："
            "grasped（两指都碰到物体）/ empty（指间是空的）/ slipped（只碰到一侧）/"
            "released（已张开且指间无物）/ still_carrying（张不开，还夹着）。"
            "判定来自接触表，不是指宽。不会移动机械臂。"
        ),
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["open", "close"]},
        }, "required": ["action"]}}},    {"type": "function", "function": {
        "name": "grasp",
        "description": (
            "在当前位姿合上夹爪抓取。内部自行处理：先张开（仿真初始是半闭的）、闭合到两指都接触物体、"
            "再继续加压到握力足够，并确认接触保持。返回概念：grasped（夹住了）/ empty（指间是空的）/ "
            "slipped（接触不稳定）。不会移动机械臂。"
        ),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "release",
        "description": (
            "张开夹爪。夹着东西时=放下物体；空手时=把夹爪张开/复位（抓取前先张开、或夹空后重新对准前都用它）。"
            "内部会等物体完全离开指间再返回（所以返回后立刻 verify 是准的）。"
            "返回概念：released（已张开）/ still_carrying（还夹着）。不会移动机械臂。"
        ),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "aim",
        "description": (
            "把一个像素画成红圈，**不移动机械臂**（干跑：不动手、不改物理状态）。"
            "返回同一张带网格的画面：红圈画在你给的那个像素上，青色十字仍然是夹爪自己。"
            "用它先确认「我说的就是这个像素」：看红圈与目标、以及红圈与十字的差距，"
            "不满意就换个像素再 aim。画面外或读不到距离的像素会被拒绝并说明原因。"
            "相机可选 agent 或 wrist（腕部更细，但腕部只能看、不能当 target 的目标）。"
        ),
        "parameters": {"type": "object", "properties": {
            "camera": {"type": "string", "enum": ["agent", "wrist"],
                       "description": "agent=全局第三人称（约 3~4mm/像素）；wrist=夹爪上的腕部相机（约 1mm/像素，只能看不能动）"},
            "u": {"type": "number", "description": "像素列，画面左上为 0"},
            "v": {"type": "number", "description": "像素行，画面左上为 0"},
        }, "required": ["u", "v"]}}},
    {"type": "function", "function": {
        "name": "target",
        "description": (
            "把像素当成**移动目标**：代码用相机标定 + 深度把该像素反投影成世界点，"
            "再加上你给的 clearance（offset_mm，世界系 xyz 毫米），然后把夹爪伺服过去。"
            "不再需要物体名或物体坐标。它只走一个**短动作块**（最多 8 个修正步，想更谨慎就传 1~3），"
            "走完就返回：没到位就用同一像素再 target 一次（新画面里会用十字告诉你爪子到哪了）。"
            "返回概念：arrived（到位）/ short（没到位）/ blocked（被挡住）。"
        ),
        "parameters": {"type": "object", "properties": {
            "camera": {"type": "string", "enum": ["agent"],
                       "description": "只能用 agent（全局第三人称）。腕部相机装在夹爪上，固定像素=相对手恒定的点，会随夹爪一起移动，无法作为移动目标"},
            "u": {"type": "number", "description": "像素列，画面左上为 0"},
            "v": {"type": "number", "description": "像素行，画面左上为 0"},
            "offset_mm": {"type": "array", "items": {"type": "number"},
                          "description": "在反投影像素点上再加的位移（世界系 xyz，毫米）。只想悬停在表面上方就加大 z（例：[0,0,120]）。默认 [0,0,0]"},
            "max_steps": {"type": "number", "description": "本调用最多走几个修正步，1~8，默认 8：调小就是更频繁地重新看图"},
        }, "required": ["u", "v"]}}},
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

# The three-section output format is not stylistic.  MEASURED 2026-10-10 with a
# single-turn probe that fed the model the exact decision frames of the run where
# the can reached the basket (frames 7/14/20/25/30 = the inputs to turns
# 8/15/21/26/31) plus that turn's real message text, changing ONLY this paragraph:
#
#   一句话前言（旧）        -> 4/5 replies called `gripper open`
#   现状/判定/动作（本段）  -> 1/5 replies called `gripper open`
#
# And the old wording produced messages that contradicted themselves: "已夹住罐头，
# 需先抬高再移至篮子上方" followed immediately by `gripper open`.  The loop never
# ran the check (0 of 80 turns doubted the release position) because "不要写多段
# 推理" removed the step from the generation.  Cost of the three sections: ~150-250
# tokens/turn (~+3.6-6 s on the 5090), still inside MAX_NEW_TOKENS=384.
SYSTEM_PROMPT_V2 = """你是机械臂任务决策核心。目标是完成当前任务。

每次只调用一个工具。工具结果是环境测量，不是行动建议；不要假设动作一定完成。
如果动作被拒绝或只完成一部分，根据返回的状态和事件自行选择下一步。
不要发明工具、参数或并行调用。verify 的成功与否只由环境验证。

输出格式：每次回复严格按下面三段写，三段都要有，标题照拄：
现状：（这一轮画面里的事实：爪上有没有东西；夹爪正下方是什么；容器（篮子）相对爪子在哪个方向、大约多远）
判定：（如果此刻张开夹爪，手里的物体会落到哪里？会不会落进容器？）
动作：（下一步调用哪个工具、参数是多少；必须与上面的判定一致——判定说不会落进容器，就不要在这一步 open）
三段写完后，紧接写出完整的工具调用。除这三段外不要写任何其他内容。

任务结构（括号内是这一步用的工具，逐步做，不要用别的工具代替）：
张开夹爪（release，空手时它就是复位/张开）→ 对准并降到抓取高度（approach）→ 合上夹爪抓起（grasp）
→ 抬高（move +z）→ 水平移到容器上方（move）→ 张开夹爪放下（release）→ 请求验证（verify）

事实（都是测量的，不是建议）：
- **放下物体（release）必须发生在容器开口的正上方**：松手前从两张画面确认容器的口就在夹爪
  正下方、物体能直接落进去。仅仅「靠近」容器就松手，物体一定落在容器外面（实测：连续 12 次在
  容器旁边或越过容器后才松手，物体一路被推到工作空间边缘）。位置拿不准就继续用 move 微调，
  不要先松手。
- 每轮都会附带两张最新的相机画面：全局第三人称视角 + 腕部向下视角（朝下看夹爪下方）。
  不需要额外的观察动作；位置和距离只能从画面判断，动作是否到位也以画面为准。
- 本配置**不提供物体的坐标数字**（没有定位工具）。
- **夹爪状态只用概念告诉你**：`grasp` 回 `grasped / empty / slipped`，`release` 回
  `released / still_carrying`。不再给你指宽数字，也不要求你算阈值：要不要夹紧、什么时候
  真算夹住，由代码用接触表判定。`grasp` 返回 `empty` 说明两指之间是空的（需要重新对准），
  `slipped` 说明接触不稳（重试，或先抬起来再说）。
- 已经夹着东西时，不要再往桌面上下降，否则会把夹着的东西刮掉；要先横向移到目标上方。
- **`release` 返回后「爪上：空」是预期的**，不代表物体掉了。只有在你没有 `release` 的
  情况下看到「爪上：空」，才是运输中掉了。
- **`verify` 说没放好、而你已经松手**：物体常常是先骑在容器口沿上，再滑进去——这不是「掉在外面」。
  正确的收尾是 `approach(<容器名>, 15)` 在开口正上方垂直压一下（爪心 0.145，
  仍在口沿 0.151 之上），然后 `verify` 一次；一旦 verify 通过就**立刻停手**。
  **不要再把夹爪降到口沿以下**（口沿以下属于容器内部），也**不要 `approach(物体)` 去看它在哪**：
  它若已在容器里，下降的指尖会顶上它、把它拨出去（实测把已落进容器的罐子顶出 10~29cm，
  把本来已经完成的任务弄成失败，并在剩下的 70 轮里再也放不回去）。
- 结果里写「未到达 / 卡住 / 只走了 X cm」都表示那个动作没有达成；以实测数字为准。

坐标系：世界坐标中 +z 向上，位置单位米，move 距离单位厘米。
工具职责：
- approach：把夹爪移到目标上方并停在指定高度（不会抬起物体）。
- move：执行明确的相对平移（抬高/降低只能用它）。
- rotate：执行明确的姿态旋转。
- grasp：在当前位姿合上夹爪抓取，只回报概念（grasped / empty / slipped）。
- release：张开夹爪放下物体，只回报概念（released / still_carrying）。
- verify：让环境检查任务是否完成。
"""


# Short note appended in `pixels` mode, ON TOP of SYSTEM_PROMPT_V2_PIXELS. It exists
# only because the shared skill file (`skills/scene_tools.md`) is still written
# around `approach(<名字>)`; the pixel workflow itself lives in the dedicated
# system prompt, not here.
PIXELS_APPENDIX = """

## 附：本模式的词汇换算
- 随附的场景事实里提到的 `approach(<名字>, N)` 在本模式下**不存在**；请一律用像素表达同一个
  意图：先 `aim(camera, u, v)` 确认（只画红圈、不动手），再 `target(camera, u, v, offset_mm)`
  移过去（一个短动作块）。
- 提到的「指宽」也不再给你了：夹爪状态只用概念告诉你（`grasp` → grasped/empty/slipped）。
"""


# `pixels` mode's own system prompt (档 B). It REPLACES the v2 prompt rather than
# extending it, and the first real pixel run is the reason: with the pixel rules
# appended below a prompt whose task structure says `approach(<名字>)`, the model
# followed the OLD recipe -- 60 turns of `move -z` blind pressing and 10 `grasp`
# pokes, 19/19 `goto_pixel` calls from the third-person view, not one from the
# wrist view that the appendix named as the millimetre stage. A prompt is a
# recipe; an appendix to a contradicting recipe is not read.
SYSTEM_PROMPT_V2_PIXELS = """你是机械臂任务决策核心。目标是完成当前任务。

每次只调用一个工具。工具结果是环境测量，不是行动建议；不要假设动作一定完成。
如果动作被拒绝或只完成一部分，根据返回的状态和事件自行选择下一步。
不要发明工具、参数或并行调用。verify 的成功与否只由环境验证。

**本回合没有任何物体坐标**：没有定位工具，也不能按物体名对准。你只能*看*画面，
再用像素说话。像素工具有两个，**先 aim 确认，再 target 移动**：
- `aim(camera, u, v)`：把**红圈**画在你说的像素上，**不动手**。用来先确认"我指的就是这个"。
- `target(camera, u, v, offset_mm, max_steps)`：真的把夹爪移过去（一个**短动作块**，最多 8 个修正步，走完就返回）。

**你自己的绝对像素读数有系统性偏差**（实测偏 30~50 像素 = 10~13cm，反复读也一样），
所以**不要指望一次算准**：aim 之后看红圈离目标差几个格子，就修几个格子，再 aim；
满意了再 target。

**每轮最多给你三张图**：全局第三人称（agent）、腕部（wrist）、以及一张**放大图**
（围绕红圈（aim）或夹爪十字（target）裁 64 像素再放大 4 倍）。三张图都是同一时刻的。
判断"准不准"用**放大图**：它把 1 个像素变成 4 个，所以 5 像素的误差在那里就是 20 像素，看一眼就知道。

**对准的判据（本模式最要紧的一条）**：
- 目标是"**红圈中心落在物体轮廓内部、四周留边差不多宽**"。红圈**碰到**物体不算对准：
  这个罐子只有约 16 像素宽（~6cm），所以"红圈压住物体左边一点"就已经很偏了。
- 但**不要追求完美**：实测抓取容差是**偏 2.2cm（约 6 像素）仍能夹住，偏 4.6cm 才夹空**。
  红圈中心已经在物体内部时就可以 `target` 去试；`grasp` 返回 `empty` 再回来重新 aim。
- **每次 aim 都花掉一个回合**，而且**同一个像素重复 aim 不会有任何新信息**（画面没变）。
  所以 aim 之后必须做出决定：要么 target，要么只改 3~5 个像素再 aim，不要原地重复。
- 红圈有一边露在物体外面，就按露出的方向修正（露右侧就减 u，露下方就减 v），再 aim 看。
- 拿不准物体到底在哪时，用腕部画面（约 1mm/像素）只看**方向**，执行靠 aim/target。

输出格式：每次回复严格按下面三段写，三段都要有，标题照抄：
现状：（画面里的事实：青色十字是夹爪自己、红圈是你说的那个点；目标在它们的哪个方向、隔几个细格；爪上有没有东西）
判定：（**先说红圈/十字落在哪个物体上（按外观说，例如"蓝标签罐"），它是不是任务目标**；
  再看放大图：红圈中心是不是在目标内部、四边留边是否均匀？不足就说清缺哪边、差几格；够不够 clearance）
动作：（下一步调用哪个工具、参数是多少；必须与上面的判定一致）
三段写完后，紧接写出完整的工具调用。除这三段外不要写任何其他内容。

任务结构（括号内是这一步用的工具）：
**先按下面「物体长什么样」认出目标容器**——场景里有好几个外形相似的罐子，
凭包装常识猜会错（实测两轮共 16 次抓取 0 次抓对）
→ 张开夹爪（release）→ aim 目标像素并调准（aim ×1~3，看放大图）→ target 到它上方
（offset z 给大一点）→ **再 aim 同一像素复核（或改像素再 aim/target）直到红圈中心在物体上**
→ **用 target 把 offset 调小（如 z=12mm）直接压到抓取高度** → 合上夹爪（grasp）
→ 抬高（move +z）→ aim+target 到容器开口上方 → 张开夹爪（release）→ 请求验证（verify）

（`move` 只用于抬高、搬运、撤离这类明确的世界轴平移；**不要在悬停后靠 `move +y/-z` 盲走对准**
—— 你看不见爪子下方，实测盲走会把它推走。）

## 怎么读画面（这是本模式唯一的位置来源）
- 画面有两种线：**细线每 8 像素一条**（用来数格子），**带数字的粗线每 32 像素一条**
  （用来定位）。例：目标在竖线 64 右边 2 个细格、横线 128 上方 1 个细格 ⇒ u≈80, v≈120。
- **放大图里同样的细线就是 32 像素一格**（放大了 4 倍），在那里数格子精 4 倍。
- **青色十字 = 夹爪自己的像素位置**；**红圈 = 你 aim 的那个像素**。
  红圈和十字重合，说明你指的正是爪子现在的位置；不重合，就把红圈挪到目标上再 target。
- 第三人称精度实测 **1 像素 ≈ 3.7mm**（偏 5 个像素就是 1.9cm）。所以它只够做**粗对准**。
- **腕部相机只能用来"看"**（`aim(camera=wrist)` 可以，`target(camera=wrist)` 会被拒绝）：
  它装在夹爪上，固定像素=相对手恒定的点，会跟着夹爪一起移动，永远追不上。它只能**判断方向**，
  执行靠 move。腕部画面实测（距目标 12cm 时，2cm 位移）：**+y → 向左 34 像素，
  -y → 向右 34 像素；+x → 向上 34 像素，-x → 向下 33 像素；+z（抬高）→ 向下 25 像素**。
  ⇒ 看目标在十字的哪一侧，就沿那个轴走 2~5cm，然后再看一眼；**不要一次算准**
  （每像素多少毫米随深度变），走一步看一步通常 1~3 步就到位。
- `offset_mm` 是"在反投影落点上再加的世界系位移（毫米）"。**落点在你看到的那层表面**，
  所以想悬在它上方就传 `[0,0,150]`（上方 15cm）。
  **两段式**：先用大 offset 悬停到上方，再用上面的腕部判断 + move 小步逼近。
  只给一次大 offset 然后改用 `move -z` 硬压是错的：你看不见爪子下方，盲压会把物体推走。
- `max_steps` 是"这次最多走几个修正步"（1~8，默认 8）。返回 `short` 表示这一小段走完还没到：
  **用同一个像素再 target 一次**即可，不是失败，也不是要换目标。

## 事实（都是测量的，不是建议）
- `grasp` 返回 `empty`（没夹到）时，说明两指之间是空的：**回到上一段用像素重新对准**，
  不要继续 `move -z` 盲压，也不要在原地反复 grasp。
- `grasp` 返回 `grasped` 之后才抬高；抬高用 `move +z`（单次上限 15cm，需要更高就再走一次）。
  横移前爪心要 ≥0.22：**手里物体底部在爪心下方约 5cm，容器口沿约 z=0.151**。
- `release` 返回后「爪上：空」是预期的。只有在你没有 `release` 的情况下看到「爪上：空」，
  才是运输中掉了。
- `verify` 说没放好、而你已经松手：物体常常是先骑在容器口沿上再滑进去。用同一条像素
  把 offset 调成 145mm 左右再压一下（爪心 0.145，仍在口沿 0.151 之上），然后 verify；
  一旦通过就**立刻停手**。不要沉到口沿以下，也不要用像素去"找"已经进容器的物体。
- 结果里写「未到达 / 卡住 / 没到位」都表示那个动作没有达成；以实测数字为准。

坐标系：世界坐标中 +z 向上，位置单位米，move 距离单位厘米，offset_mm 单位毫米。
工具职责：
- aim：把像素画成红圈，**不移动机械臂**；用来确认自己指对了（可用 agent 或 wrist 画面），
  返回时附带围绕红圈的放大图。
- target：把像素（+ 你给的 offset）作为目标伺服过去，只走一个短动作块；只能用 agent 画面，
  返回时附带围绕夹爪十字的放大图（用来看爪子离目标还差多少）。
- move：执行明确的相对平移（抬高/降低只能用它）。
- rotate：执行明确的姿态旋转。
- grasp：合上夹爪抓取，只回报概念（grasped / empty / slipped）。
- release：张开夹爪（空手时就是复位/张开），只回报概念（released / still_carrying）。
- verify：让环境检查任务是否完成。
"""


PROMPT_MODES = {
    "full": SYSTEM_PROMPT,
    "plain": SYSTEM_PROMPT_PLAIN,
    "v2": SYSTEM_PROMPT_V2,
}


# `look` mode's own schema: ONE camera at a time, and it is the only way to see anything.
# The v2 `observe` schema is a three-in-one tool (object coordinates / frames / state) and
# two of its three branches are exactly what this mode removes.
LOOK_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "look",
        "description": (
            "看一次相机，返回一张新画面。**不移动机械臂。**"
            "camera=agent 是全局第三人称视角（看位置关系、看它现在在哪），"
            "camera=wrist 是腕部俯视（看细节、看夹爪正下方有什么）。"
            "**本模式不会再自动给你图**：你看到的每一张画都是你自己 look 出来的，"
            "而且它**会留在对话里**（后面的回合还能回头对比'走这步之前/之后'）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera": {"type": "string", "enum": ["agent", "wrist"],
                           "description": "要看哪个相机；省略则为 agent"},
            },
            "required": [],
        },
    },
}


LOOKS_APPENDIX = """

## 附：本模式的词汇换算
- 随附的场景事实里的 `approach(<名字>)` / `aim` / `target` 在本模式下**不存在**；
  `grasp()` / `release()` 在本模式下就是 `gripper close` / `gripper open`。
- 本模式**不会自动给你画面**。要看就 `look`。你不看的时候手里只有数字。
"""


# `look` mode's system prompt. It is the `raw` prompt minus the promise of frames, plus
# the one thing the model has to understand: seeing is now an ACTION it takes, and the
# frames it takes stay in the conversation, so "did that step take me past it?" is now a
# question it can actually answer by comparing two frames it asked for.
SYSTEM_PROMPT_V2_LOOK = """你是机械臂任务决策核心。目标是完成当前任务。

每次只调用一个工具。工具结果是环境测量，不是行动建议；不要假设动作一定完成。
不要发明工具、参数或并行调用。verify 的成功与否只由环境验证。

**本模式只有四个工具**：`look`（看相机，唯一的信息来源）、`move`（世界轴平移，
唯一能改变高度的手段）、`gripper`（open/close，只回报概念）、`verify`（问环境完成没有）。

**你不会自动获得任何画面。** 想看就自己 `look`：

- 想知道“东西现在在哪儿、我走到哪了” → `look(agent)`；
- 想知道“夹爪正下方是什么” → `look(wrist)`。

**你 look 出来的画面会留在对话里**，所以有两种用法：一直 look，也可以走几步再 look
一次、然后**回头对比“走之前”和“走之后”两张图**——这就是你判断自己一步走了多远、
会不会过头的手段。宁可多 look 一次，也不要盲走。

输出格式：每次回复严格按下面三段写，三段都要有，标题照抄：
现状：（你最近看到的画面里的事实，或“本轮没看图”；不要编造你没看过的画面）
判定：（目标相对爪心偏哪个方向、你打算怎么动；或该开/关夹爪）
动作：（下一步调用哪个工具、参数是多少；必须与上面的判定一致）
三段写完后，紧接写出完整的工具调用。除这三段外不要写任何其他内容。

任务结构（括号内是这一步用的工具）：
先 `look(agent)` 找到目标 → `gripper open` 张开夹爪 → 用小步 `move` 靠近它，
**每走到不确定的时候先 look 再看** → `look(wrist)` 确认夹爪就在它上方 →
`move -z` 降到刚好碰到它（降不动了就停）→ `gripper close` →
看回报是「夹住了」还是「夹空了」：夹空就 `move +z` 抬起 5cm、重新对齐再试 →
`move +z` 抬到 0.25 以上 → 移动到容器上方（到了先 `look` 确认）→ 降到合适高度 →
`gripper open` → 抬高一点 → `verify`

## 事实（都是测量的，不是建议）
- `gripper close` 只回概念：「夹住了」= 两指都碰到了物体；「夹空了」= 指间是空的；
  「接触不稳」= 只碰到一侧。**夹空时不要原地再关，先抬起再重新对准。**
- 抬高/降低只能用 `move ±z`；单次上限 15cm。横移前爪心要 ≥0.22，因为手里物体底部在爪心下方约 5cm。
- 结果里写「未到达 / 只走了 X cm / ⚠」都表示那个动作没有完全达成；以实测数字为准。

坐标系：世界坐标中 +z 向上，位置单位米。
"""


def get_system_prompt(
    mode: str = "full", observation_mode: str = "truth", move_ladder: bool = False
) -> str:
    """Pick the system prompt. Unknown modes fall back to `full`.

    档 B: `pixels` mode gets its OWN v2 prompt instead of the name-based one, because
    the first real pixel run showed the model following the old recipe (blind
    `move -z` + repeated `grasp`, never the wrist view) while the pixel rules sat in
    an appendix underneath that recipe -- and a prompt IS a recipe.

    档 D: the same lesson applied to the ladder. `raw`'s recipe says "一次只走一小段
    （1~5cm）", which the ladder CANNOT express -- the available steps are 1cm and 10cm --
    so leaving it in place left the model choosing `far` to satisfy "at least the 4cm it
    says it needs" (MEASURED: 4.07cm error -> far, 2.46x overshoot, six times in a row).
    The step rule is therefore SWAPPED, not appended to.
    """
    mode_l = str(mode).strip().lower()
    mode_o = str(observation_mode).strip().lower()
    if mode_o == "pixels" and mode_l == "v2":
        return SYSTEM_PROMPT_V2_PIXELS
    if mode_o == "raw" and mode_l == "v2":
        return raw_prompt(move_ladder)
    if mode_o == "look" and mode_l == "v2":
        return SYSTEM_PROMPT_V2_LOOK
    return PROMPT_MODES.get(mode_l, SYSTEM_PROMPT)


# The two sentences in the raw prompt that the ladder must REPLACE. Kept as their own
# constants so the swap is a one-line lookup and a grep for either wording finds both
# the rule and its ladder-aware version.
RAW_NO_ARITHMETIC = "全靠你自己换算"
RAW_NO_ARITHMETIC_LADDER = (
    "全靠你自己判断；**但距离不用你算——只有 far(10cm) / near(1cm) / fine(1mm) 三档**"
)
RAW_STEP_RULE = "**一次只走一小段（1~5cm），然后重新看**。"
RAW_STEP_RULE_LADDER = """**一次只走一档，然后重新看**。你只报「哪个轴 + 还差几像素」（`gap_px`），厘米数由代码算：

| 你量的像素差 | 代码取到的档位 |
| --- | --- |
| ≥ 32 像素 | `far`（10cm） |
| 3 ~ 32 像素 | `near`（1cm） |
| < 3 像素 | `fine`（1mm） |

**不要自己把像素换算成厘米。**实测：模型把「75像素 × 3.8mm」算成 2.8cm（真值 28cm）、
把 10~19cm 的真差当成 2~3cm，于是永远报小、永远走 `near`、80 回合到不了。
你只要**诚实量两个东西**：目标在十字的哪一侧、差多少像素。
每步结果里会告诉你这一步实际移动了多少像素，用它校正：走完目标跑到十字另一边=报大了；
几乎没动=报小了。"""


# Mode-specific appendix. It exists because the SHARED skill file
# (`skills/scene_tools.md`) is injected into EVERY mode's system prompt and it
# describes the other modes' tools; the dedicated prompt states the real surface, and
# this is the one-line correction for the vocabulary the skill file uses.
RAWS_APPENDIX = """

## 附：本模式的词汇换算
- 随附的场景事实里提到的 `approach(<名字>)` / `aim` / `target` **在本模式下不存在**。
  本模式只有三个工具：`move`（世界轴平移）、`gripper`（开/关）、`verify`（让环境判定）。
- 提到的「指宽」也不会给你数字：`gripper` 只回报概念（夹住了 / 夹空了 / 接触不稳 / 已张开 / 还夹着）。
"""


def mode_appendix(observation_mode: str) -> str:
    """The vocabulary correction for this mode's skill-file mismatch (may be empty)."""
    mode = str(observation_mode).strip().lower()
    if mode == "pixels":
        return PIXELS_APPENDIX
    if mode == "raw":
        return RAWS_APPENDIX
    if mode == "look":
        return LOOKS_APPENDIX
    return ""


def ladder_appendix(move_ladder: bool) -> str:
    """The distance-ladder instructions, appended only when the ladder is on."""
    return MOVE_LADDER_APPENDIX if move_ladder else ""


# `raw` mode's own system prompt. It is the LONGEST tool description in the project
# precisely because the surface is the smallest: with no localisation aid, the only
# thing the harness can still give the model is the measured relationship between what
# it sees and how the world moves -- and that relationship is measurement, not a
# decision about what to do.
SYSTEM_PROMPT_V2_RAW = """你是机械臂任务决策核心。目标是完成当前任务。

每次只调用一个工具。工具结果是环境测量，不是行动建议；不要假设动作一定完成。
不要发明工具、参数或并行调用。verify 的成功与否只由环境验证。

**本模式只有三个工具**：`move`（世界轴平移，唯一能改变高度的手段）、`gripper`（open/close，
只回报概念）、`verify`（问环境任务完成没有）。
**每轮自动给你两张图**：全局第三人称（agent）与腕部（wrist）。
没有任何按名字或按像素对准的工具：**从"我看到什么"到"该往哪个世界轴走多少厘米"，
全靠你自己换算**。

## 怎么把画面换成移动（这是本模式唯一的位置来源）
- 图上的坐标系：+x = 靠近相机（画面更靠下）；-x = 远离相机（画面更靠上）；
  +y = 画面更靠右；-y = 画面更靠左；+z = 向上。
- 尺度：全局画面里物体处约 **1 像素 ≈ 3~4mm**（随远近变），所以 **移动 2cm ≈ 画面移动 5~7 像素**。
  腕部画面近得多（约 1 像素 ≈ 1mm），适合看细节与对称性，但它离物体越近，同样的像素差对应的实际距离越小。
- **一次只走一小段（1~5cm），然后重新看**。本模式的实验已经证明：不走一步看一眼而是
  连续盲走，会把东西推走或夹空。走多了一步看起来"没动"就是碰到东西了，换一个轴试。
- 你只知道你自己的爪心位置（每轮状态行里的世界坐标，单位米）。物体在哪里必须从画面判断。

输出格式：每次回复严格按下面三段写，三段都要有，标题照抄：
现状：（画面里的事实：目标在画面的哪个位置、爪心在画面的哪个位置、它们相差几个像素；爪上有没有东西）
判定：（目标相对爪心偏哪个方向、大约多少像素 ⇒ 换算成哪个世界轴、走多少厘米；或该开/关夹爪）
动作：（下一步调用哪个工具、参数是多少；必须与上面的判定一致）
三段写完后，紧接写出完整的工具调用。除这三段外不要写任何其他内容。

任务结构（括号内是这一步用的工具）：
张开夹爪（gripper open）→ 按外观认出目标（见场景事实）→ **分几次小步 move，把爪心对到目标正上方**
（每次 move 后重新看图）→ `move -z` 降到刚好碰到它（降不动了就停，不要再压）→ `gripper close`
→ 看回报是「夹住了」还是「夹空了」：夹空就抬起 5cm、重新对准再试 → `move +z` 抬高到 0.25 以上
→ 小步 move 到容器上方 → 降到你估计的合适高度 → `gripper open` → 抬高一点 → `verify`

## 事实（都是测量的，不是建议）
- `gripper close` 只回概念：**「夹住了」= 两指都碰到了物体**；「夹空了」= 指间是空的；
  「接触不稳」= 只碰到一侧。**夹空时不要原地再关，先抬起再重新对准。**
- `gripper open` 之后「爪上：空」是预期的（你没让它夹东西时它本来就是空的）。
- 抬高/降低只能用 `move ±z`；单次上限 15cm。横移前爪心要 ≥0.22，因为手里物体底部在爪心下方约 5cm。
- 容器口沿约 z=0.151，罐子高约 8.5cm。放手后物体可能先骑在口沿上，`verify` 说没好时
  可以在容器上方轻轻再压一下（爪心不要低于 0.145），再 verify 一次；一旦通过就**立刻停手**。
- 结果里写「未到达 / 只走了 X cm / ⚠」都表示那个动作没有完全达成；以实测数字为准。

坐标系：世界坐标中 +z 向上，位置单位米，move 距离单位厘米。
"""


def raw_prompt(move_ladder: bool = False) -> str:
    """The `raw` system prompt, with the step rule matching the magnitude menu."""
    if not move_ladder:
        return SYSTEM_PROMPT_V2_RAW
    return (
        SYSTEM_PROMPT_V2_RAW
        .replace(RAW_NO_ARITHMETIC, RAW_NO_ARITHMETIC_LADDER)
        .replace(RAW_STEP_RULE, RAW_STEP_RULE_LADDER)
    )


def get_tool_schemas(
    protocol: str = "v2", observation_mode: str = "truth", move_ladder: bool = False
) -> list[dict[str, Any]]:
    """The tool surface actually advertised to the model.

    A WHITELIST per observation mode (`tool_protocol.MODE_TOOLS`). Schemas for every
    tool stay in this file so re-enabling one is a one-line change in that table;
    what the model sees is what its mode can actually honour. The `raw` mode keeps
    exactly `move`, `gripper`, `verify` -- no localisation aid by name or by pixel.

    With `move_ladder`, the `move` entry is swapped for the version whose magnitude is
    a NAMED rung (档 D). Nothing else changes -- the ladder is about how big a step is,
    not about which tools exist.
    """
    schemas = TOOL_SCHEMAS_V2 if str(protocol).strip().lower() == "v2" else LEGACY_TOOL_SCHEMAS
    if move_ladder:
        schemas = [MOVE_LADDER_SCHEMA if s["function"]["name"] == "move" else s for s in schemas]
    if str(observation_mode).strip().lower() == "look":
        # `look` replaces the three-in-one `observe`: one camera per call, and the only
        # source of pixels. Names, positions and object coordinates stay absent.
        schemas = [LOOK_SCHEMA if s["function"]["name"] == "observe" else s for s in schemas]
    if str(protocol).strip().lower() != "v2":
        return schemas
    surface = tools_for_mode(observation_mode)
    return [s for s in schemas if canonical_name(s["function"]["name"]) in surface]
