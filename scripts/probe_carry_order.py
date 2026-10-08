"""Is the ORDER of the L-moves what matters (i.e. WHERE the turn happens)?

Evidence so far
---------------
    +x15          -> HELD      (ends at x +15, stays away from the base)
    +y15          -> HELD      (ends at y +15; this leg IS along the opening axis)
    +x15 +y15     -> DROPPED
    interleaved   -> DROPPED
    cap 1.0 global-> 0/5 HELD  (every grasp closed on air, so that route is dead)
    gentle(1cm)   -> 1/5 HELD  (grips fine, still sheds the can)

Both single-axis carries are safe, so "along the opening axis" alone does not
explain the failures. New hypothesis: after +x15 the EEF sits at x ~ +0.03,
almost directly over the robot base (a near-singular pose for the Panda), and a
lateral move FROM THAT POSE is what sheds the can -- the same lateral move at
x = -0.12 is fine.

Test: the same two legs in BOTH orders, same endpoint, repeated.
    +x15+y15 : x first  (turn at x ~ +0.03)
    +y15+x15 : y first  (turn at x ~ -0.12)
If only the x-first order fails, the driver is the POSE AT THE TURN, not the
turn itself -- and the fix is to plan the path so the lateral leg happens away
from the base (or to re-orient before turning).

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_carry_order.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

REPEATS = 5
PATHS = {
    # Baseline: reach (15, 0) and stop. HELD 4/5 before.
    "+x15 only": [("+z", 5), ("+z", 5), ("+x", 5), ("+x", 5), ("+x", 5)],
    # Reach (15, 0), then keep going UP. Isolates "is the POSE at x~+0.03 bad"
    # from "is any motion from there bad" -- and it cannot hit anything.
    "+x15 then +z10": [
        ("+z", 5), ("+z", 5), ("+x", 5), ("+x", 5), ("+x", 5),
        ("+z", 5), ("+z", 5),
    ],
    # Reach (15, 0), then come straight back. If this HELDs, being AT x~+0.03 is
    # fine and only the *turn into +y* matters; if it drops, that pose is bad.
    "+x15 then -x15": [
        ("+z", 5), ("+z", 5), ("+x", 5), ("+x", 5), ("+x", 5),
        ("-x", 5), ("-x", 5), ("-x", 5),
    ],
    # Control for the return leg on its own axis.
    "-x15 only": [("+z", 5), ("+z", 5), ("-x", 5), ("-x", 5), ("-x", 5)],
}


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    for label, actions in PATHS.items():
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
            for direction, dist in actions:
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
