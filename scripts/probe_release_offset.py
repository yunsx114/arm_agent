"""Which (carry height, release offset) actually puts the can in the basket?

MEASURED FACTS (2026-10-11):
  * basket_1_main geom z ∈ [-0.0214, +0.1513]: the rim is at 0.1513, NOT the
    0.115 the skill assumed.  The basket body's reported z is -0.0046.
  * alphabet_soup_1_main is 0.0852 m tall; the gripper holds it at mid-height, so
    the can's bottom rides ~0.050 below the EEF.
  * `descend_to` corrects XY BEFORE each descent step, so the lateral travel runs
    at whatever z the arm is already at.

PROBE BUG FOUND THE HARD WAY (first two sweeps, all 18 rows "failed"): the sim
starts with the gripper HALF CLOSED (finger width 0.0207), which is why every
episode's first action is `gripper open`.  Without opening first, `close` only
pinches (0.0183), `_carrying()` still reports True, and the can is shed at the
very start of the lateral move -- identically for every parameter, which is what
made the first sweeps look parameter-independent.  This version opens first.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_release_offset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

RAISES_CM = [15, 20, 25]
OFFSETS_CM = [20, 25, 30]
CAN_BOTTOM_OFFSET = 0.050


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)
    n_act = int(sim.env.env.action_dim)
    print(f"{'raise':>6} {'offset':>7} {'close_w':>8} {'eef_z':>8} {'can_bot':>8} "
          f"{'can_end z':>10} {'can_end xy':>21} {'basket_mm':>10} success")
    print("-" * 106)
    for raise_cm in RAISES_CM:
        for off in OFFSETS_CM:
            obs = sim.reset(init_state_idx=0)
            can = np.asarray(obs["objects"]["alphabet_soup_1"], dtype=float)
            basket = np.asarray(obs["objects"]["basket_1"], dtype=float)

            adapter.set_gripper("open")                     # <- the fix
            adapter.descend_to((can[0], can[1]), can[2] + 0.012)
            adapter.set_gripper("close")
            close_w = adapter._gripper_now()
            carried = adapter._carrying()
            adapter.move("+z", float(raise_cm))
            adapter.descend_to((basket[0], basket[1]), basket[2] + off / 100.0)
            eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
            still = adapter._carrying()
            adapter.set_gripper("open")
            for _ in range(25):
                sim.step(np.zeros(n_act))

            after = sim.pack(sim._obs)["objects"]
            ce = np.asarray(after["alphabet_soup_1"], dtype=float)
            be = np.asarray(after["basket_1"], dtype=float)
            moved = float(np.linalg.norm(be[:2] - basket[:2])) * 1000.0
            print(f"{raise_cm:>4}cm {off:>6}cm {close_w:>8.4f} {eef[2]:>8.4f} "
                  f"{eef[2] - CAN_BOTTOM_OFFSET:>8.4f} {ce[2]:>10.4f} "
                  f"({ce[0]:+.4f},{ce[1]:+.4f}) {moved:>8.1f}mm {bool(sim.check_success())}"
                  + ("" if carried and still else f"   [中途掉了 carry={still}]"))
    sim.close()
    return 0


raise SystemExit(main())
