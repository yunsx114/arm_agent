"""Why did `open` fail to release the can? (v3 declare_done failed twice)

MEASURED (episode_000_1791439583):
    t41 descend_to(basket_1, +15) -> eef (0.0141, 0.2701, 0.1432)   # correct
    t42 set_gripper open          -> width 0.0312 (holding)  5 ticks
    t43 move +z10                 -> eef z 0.2397
    t46 locate can                -> can z 0.2274   # can RODE UP with the hand
So `open` left the pads at 0.0312 (not the ~0.0388 of a free gripper), i.e. the
can was never released, and declare_done correctly refused.

Hypothesis: with an object between the pads, `set_gripper(open)` stops after
only a few ticks because the convergence test (|dw| < eps for 5 ticks) trips
while the pads are still pressing on the object.

This probe reproduces it and prints the tick-by-tick width.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_open_release.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import arm_agent.sim.action_adapter as A  # noqa: E402
from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402


def trace_gripper(adapter: ActionAdapter, action_name: str, max_steps: int = 140) -> list[float]:
    """Run the raw open/close command, recording width every tick."""
    sign = -1.0 if action_name == "open" else 1.0
    widths = [adapter._gripper_now()]
    for _ in range(max_steps):
        a = np.zeros(7)
        a[6] = sign
        adapter.sim.step(a)
        widths.append(adapter._gripper_now())
        if len(widths) > 12 and abs(widths[-1] - widths[-8]) < 1e-6:
            break
    return widths


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)
    obs = sim.reset(init_state_idx=0)
    key = [k for k in obs["objects"] if "soup" in k][0]
    cx, cy, cz = obs["objects"][key]

    adapter.descend_to((cx, cy), cz + 0.012)
    c = adapter.set_gripper("close")
    print(f"close : {c.note}  ({c.steps} ticks)")

    # 1) what the ADAPTER's open does
    o = adapter.set_gripper("open")
    print(f"open  : {o.note}  ({o.steps} ticks)")
    print(f"   can still on hand? |eef-can| = "
          f"{float(np.linalg.norm(np.asarray(sim.pack(sim._obs)['objects'][key]) - np.asarray(sim._obs['robot0_eef_pos'])))*1000:.1f} mm")

    # 2) what the RAW command would do if let run to completion
    sim.reset(init_state_idx=0)
    adapter.descend_to((cx, cy), cz + 0.012)
    adapter.set_gripper("close")
    widths = trace_gripper(adapter, "open", max_steps=60)
    print("\nraw open width trace (every 5th tick):")
    for i in range(0, len(widths), 5):
        print(f"   tick{i:>3}: {widths[i]:.4f}")
    print(f"   final : {widths[-1]:.4f}  (free gripper opens to ~0.0388)")
    print(f"   GRIPPER_OPEN_WIDTH={A.GRIPPER_OPEN_WIDTH} GRIPPER_EMPTY_WIDTH={A.GRIPPER_EMPTY_WIDTH}"
          f" CONVERGE_EPS={A.GRIPPER_CONVERGE_EPS} SETTLE_AVG={A.GRIPPER_SETTLE_AVG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
