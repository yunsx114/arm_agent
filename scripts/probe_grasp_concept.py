"""Does the grasp/release CONCEPT agree with the physics? (档 A validation)

WHY: 档 A replaces `gripper(open|close)` + a finger-width readout with two
primitives that answer in concepts (`grasped / empty / slipped`, `released /
still_carrying`). The claim behind that is "contact is the fact, the width is a
proxy" -- so this probe checks the claim on the real sim, including the case the
width rule gets WRONG, and it also measures whether `release()`'s settle window
makes an immediately following `check_success()` honest (the systematic false
negative that sent the model diving back into the basket in 8/8 episodes).

Scenarios:
  1. reset            -> pads half closed, NO contact (old width rule says "carrying")
  2. grasp() in air   -> empty
  3. grasp() on can   -> grasped (both sides, one body)
  4. grasp() off-axis -> slipped/empty (only one pad can reach the can)
  5. release() over basket -> released, and check_success() right after vs settled

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_grasp_concept.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402


def verdict(adapter: ActionAdapter) -> tuple[bool, set[str], str | None]:
    body, sides = adapter.held_object()
    return len(sides) == 2, sides, body


def old_width_rule(adapter: ActionAdapter) -> str:
    """What the OLD width test would have concluded (for comparison only)."""
    width = adapter._gripper_now()
    if width >= 0.034:
        return "张开"
    if width <= 0.006:
        return "闭合（空）"
    return "半闭（指间有物体）← 旧规则会当成夹着"


def show(label: str, adapter: ActionAdapter) -> None:
    held, sides, body = verdict(adapter)
    width = adapter._gripper_now()
    print(f"  {label:34s} 指宽={width * 1000:6.2f}mm  接触={sorted(sides) or '无'} "
          f"双向={held}  物体={body}  旧规则={old_width_rule(adapter)}")


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)
    n_act = int(sim.env.env.action_dim)

    obs = sim.reset(init_state_idx=0)
    print("\n=== 1. reset（仿真启动状态）===")
    show("reset", adapter)

    can = np.asarray(obs["objects"]["alphabet_soup_1"], dtype=float)
    basket = np.asarray(obs["objects"]["basket_1"], dtype=float)

    print("\n=== 2. 空中 grasp()（指间什么都没有）===")
    out = adapter.grasp()
    show("grasp()", adapter)
    print(f"  → status={out.status}  measurement={out.measurement}")

    print("\n=== 3. 正确位置 grasp()（先张开 → 下降对准 → 合上）===")
    adapter.move("+z", 10.0)
    adapter.release()          # 空手也要先张开：闭着两指下降会把物体顶走（已实测）
    adapter.descend_to((can[0], can[1]), can[2] + 0.012)
    show("下降对准后", adapter)
    out = adapter.grasp()
    show("grasp()", adapter)
    print(f"  → status={out.status}  measurement={out.measurement}")

    print("\n=== 4. 抬起后 release()（高空松开）===")
    adapter.move("+z", 15.0)
    show("抬起后", adapter)
    out = adapter.release()
    show("release()", adapter)
    print(f"  → status={out.status}  measurement={out.measurement}")

    print("\n=== 5. 偏轴 grasp()（只可能一侧碰到罐子）===")
    obs2 = sim.reset(init_state_idx=0)
    can = np.asarray(obs2["objects"]["alphabet_soup_1"], dtype=float)
    basket = np.asarray(obs2["objects"]["basket_1"], dtype=float)
    adapter.release()
    adapter.descend_to((can[0] + 0.022, can[1]), can[2] + 0.012)
    show("偏移 2.2cm 后", adapter)
    out = adapter.grasp()
    show("grasp()", adapter)
    print(f"  → status={out.status}  measurement={out.measurement}")

    print("\n=== 6. 完整流程 + release 之后的 check_success（关键：落定窗口）===")
    obs3 = sim.reset(init_state_idx=0)
    can = np.asarray(obs3["objects"]["alphabet_soup_1"], dtype=float)
    basket = np.asarray(obs3["objects"]["basket_1"], dtype=float)
    adapter.release()                                     # 张开（抓取前的准备）
    adapter.descend_to((can[0], can[1]), can[2] + 0.012)
    g = adapter.grasp()
    adapter.move("+z", 25.0)
    adapter.descend_to((basket[0], basket[1]), basket[2] + 0.25)
    eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
    rel = adapter.release()
    immediate = bool(sim.check_success())
    for _ in range(25):
        sim.step(np.zeros(n_act))
    settled = bool(sim.check_success())
    after = sim.pack(sim._obs)["objects"]
    ce = np.asarray(after["alphabet_soup_1"], dtype=float)
    print(f"  grasp  → {g.status} {g.measurement}")
    print(f"  release→ {rel.status} {rel.measurement}")
    print(f"  松手时爪心 z={eef[2]:.4f}（物体底≈{eef[2] - 0.050:.4f}）")
    print(f"  check_success：release() 返回后立刻={immediate}，落定 25 tick 后={settled}")
    print(f"  罐子体心 z={ce[2]:.4f}（篮内静止≈0.0554，口沿≈0.1513）")

    sim.close()
    return 0


raise SystemExit(main())
