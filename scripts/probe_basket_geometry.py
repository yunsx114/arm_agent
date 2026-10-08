"""How high is the basket rim, and at what descent does the gripper shove it?

CONTEXT (measured, episode_000_1791437611)
    t35 descend_to(basket, z_offset_cm=10) then open
    t40 locate basket -> (+0.0148 -> -0.0100, +0.2521 -> +0.5413)   <- pushed 29 cm
    t47 locate basket -> (+0.0022, +0.6561)                          <- pushed 40 cm total
The basket is a DYNAMIC body, so driving the gripper down into it shoves it away;
the model then chased it for 28 turns until the arm hit its workspace limit.

Two measurable answers are needed:
    1. where does the gripper END UP for each z_offset (i.e. how deep it goes), and
    2. at which z_offset does the basket start moving?

Then "which carry height is safe" becomes arithmetic instead of a guess.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_basket_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

OFFSETS_CM = [4, 6, 8, 10, 12, 15, 18, 22, 26]


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    obs = sim.reset(init_state_idx=0)
    objs = obs["objects"]
    bkey = [k for k in objs if "basket" in k][0]
    b0 = np.asarray(objs[bkey], dtype=float)
    print(f"basket centre (rest): ({b0[0]:+.4f}, {b0[1]:+.4f}, {b0[2]:+.4f})")
    print(f"{'offset':>8} {'eef_end_z':>10} {'eef_xy_err':>11} {'basket_moved':>13}  note")
    print("-" * 62)

    for off_cm in OFFSETS_CM:
        obs = sim.reset(init_state_idx=0)
        b_start = np.asarray(obs["objects"][bkey], dtype=float)
        z_target = b_start[2] + off_cm / 100.0
        out = adapter.descend_to((b_start[0], b_start[1]), z_target)
        eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
        b_end = np.asarray(sim.pack(sim._obs)["objects"][bkey], dtype=float)
        moved = float(np.linalg.norm(b_end[:2] - b_start[:2])) * 1000.0
        xy_err = float(np.linalg.norm(eef[:2] - b_start[:2])) * 1000.0
        print(
            f"{off_cm:>6}cm {eef[2]:>10.4f} {xy_err:>10.1f}mm {moved:>11.1f}mm  "
            f"{out.note[:26]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
