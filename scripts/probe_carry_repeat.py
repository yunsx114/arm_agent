"""Does gentle really hold, or was that one HELD run luck?

MEASURED spread: |eef-can| right after a grasp ranges 13-26 mm across resets, and
that spread ALONE flips the carry outcome. A single run therefore cannot tell a
real fix from a lucky grip, so each configuration is repeated N times and scored
by survival rate.

  A) MOVE_STEP_CAP_M = 5 cm  -> the adapter's gentle mode slows carrying moves
  B) MOVE_STEP_CAP_M = 1 cm  -> the global slowdown that once tested HELD

If A and B score the same, the adapter fix is doing its job. If B wins clearly,
the slowdown has to be global -- which is not free: 2.5 cm/tick during the
DESCENT closes the pads on air, so it would need a different gate.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_carry_repeat.py
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

REPEATS = 5
L_CARRY = [
    ("+z", 5), ("+z", 5),
    ("+x", 5), ("+x", 5), ("+x", 5),
    ("+y", 5), ("+y", 5), ("+y", 5),
]
CONFIGS = {"cap5.0 (adapter gentle)": 0.05, "cap1.0 (global slow)": 0.01}


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    for label, cap in CONFIGS.items():
        A.MOVE_STEP_CAP_M = cap
        held_n = 0
        widths: list[float] = []
        rides: list[float] = []
        for _rep in range(REPEATS):
            obs = sim.reset(init_state_idx=0)
            key = [k for k in obs["objects"] if "soup" in k][0]
            cx, cy, cz = obs["objects"][key]

            adapter.descend_to((cx, cy), cz + 0.012)
            widths.append(
                float(adapter.set_gripper("close").note.split("width ")[1].split()[0])
            )
            for direction, dist in L_CARRY:
                adapter.move(direction, dist)

            can = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
            eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
            rides.append(float(np.linalg.norm(can - eef)) * 1000.0)
            held_n += int(can[2] > 0.08)

        print(f"\n{label}:  {held_n}/{REPEATS} HELD")
        print(f"   close_w : {[round(w, 4) for w in widths]}")
        print(f"   |eef-can| after carry (mm): {[round(r) for r in rides]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
