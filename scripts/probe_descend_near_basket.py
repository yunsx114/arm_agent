"""Is the basket knocked over by LUCK, or by the DESCENT PATH (systematic)?

v5 and v6 both show the basket being disturbed while the model re-grasps a
dropped can beside it. Two candidate mechanisms:

  (a) luck -- a rare unfavourable geometry, nothing to fix; or
  (b) `descend_to` always brings the pads down LOW, so it will disturb the basket
      whenever the target is within some radius of it. That is SYSTEMATIC and
      fixable (approach from above, keep a safe clearance).

Test: descend to points at a range of distances from the basket (same z target,
i.e. table height) and report how far the basket moves. A sharp distance
threshold means (b): the behaviour is geometric, not random.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_descend_near_basket.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

DX_CM = [0, 2, 4, 6, 8, 10, 14, 18, 24]


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    print(f"{'dx_from_basket':>15} {'basket_moved':>13} {'basket_dz':>10} {'eef_end_z':>10}  note")
    print("-" * 74)
    for dx_cm in DX_CM:
        obs = sim.reset(init_state_idx=0)
        objs = obs["objects"]
        bkey = [k for k in objs if "basket" in k][0]
        b0 = np.asarray(objs[bkey], dtype=float)
        target = (b0[0] + dx_cm / 100.0, b0[1])  # beside the basket, same y

        out = adapter.descend_to(target, b0[2] + 0.012)  # to table height
        b1 = np.asarray(sim.pack(sim._obs)["objects"][bkey], dtype=float)
        eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
        moved = float(np.linalg.norm(b1[:2] - b0[:2])) * 1000.0
        print(
            f"{dx_cm:>13}cm {moved:>11.1f}mm {b1[2] - b0[2]:>+10.4f} {eef[2]:>10.4f}  "
            f"{out.note[:24]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
