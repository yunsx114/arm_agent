"""Fewer stop-starts vs more: does ONE long servo shed the payload less?

CONTEXT (measured, all repeats of 5)
    +z only            5/5      (3 moves)
    +x15 (3x5cm)       4/5      (5 moves)
    +y15 (3x5cm)       5/5      (5 moves)
    +x15 then +z10     2/5      (7 moves)   <- even straight UP sheds it
    +x15 then -x15     2/5      (8 moves)
    +x15+y15           1/5      (8 moves)
Every hypothesis about DIRECTION failed; what correlates with the outcome is how
many move calls (stop-starts) the carry contains. This probe tests the obvious
consequence: cover the same 15 cm in ONE servo instead of three.

Same distance, same speed inside the servo (gentle caps at 1 cm/tick while
carrying), so the only difference is the number of stop-starts: 1 vs 3.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_move_chunking.py
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

REPEATS = 6
# (label, moves, max distance per call in cm)
CONFIGS = {
    "+x15 as 3x5cm (current)": ([("+z", 5), ("+z", 5), ("+x", 5), ("+x", 5), ("+x", 5)], 5.0),
    "+x15 as 1x15cm": ([("+z", 5), ("+z", 5), ("+x", 15)], 20.0),
    "+y15 as 3x5cm": ([("+z", 5), ("+z", 5), ("+y", 5), ("+y", 5), ("+y", 5)], 5.0),
    "+y15 as 1x15cm": ([("+z", 5), ("+z", 5), ("+y", 15)], 20.0),
}


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    for label, (moves, max_cm) in CONFIGS.items():
        A.MOVE_MAX_DISTANCE_CM = max_cm
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
            for direction, dist in moves:
                adapter.move(direction, dist)

            can = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
            eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
            rides.append(float(np.linalg.norm(can - eef)) * 1000.0)
            held_n += int(can[2] > 0.08)

        pct = 100.0 * held_n / REPEATS
        print(f"\n{label:28} {held_n}/{REPEATS} HELD  ({pct:.0f}%)")
        print(f"   close_w : {[round(w, 4) for w in widths]}")
        print(f"   |eef-can| after carry (mm): {[round(r) for r in rides]}")

    A.MOVE_MAX_DISTANCE_CM = 5.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
