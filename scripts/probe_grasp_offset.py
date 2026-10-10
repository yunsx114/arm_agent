"""Which descent z actually grasps the can? Measured with the SHIPPED code path.

Why: two consecutive episodes grasped the can at finger width 0.018-0.020 (vs
0.031 when the grip is good) and shed it during the +y carries. The descent is
`descend_to(can_xy, can_z + offset)` with offset = 1.2cm (the tool's default),
and both times the servo ended ~12mm ABOVE the commanded z while pressing -- i.e.
the pads were sitting on the can's top rim and closed on the rim, not the body.

The old probe_grasp_height.py cannot answer this any more (its private go_to does
not converge against the current scene), so this uses ActionAdapter.descend_to
itself, which is the exact path the agent calls.

Run: bash scripts/simenv.sh scripts/probe_grasp_offset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

SOUP = "alphabet_soup_1"
OFFSETS_CM = [3.0, 2.0, 1.2, 0.5, 0.0, -0.5, -1.0]


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    print(f"{'offset':>7} {'eef_z':>8} {'z_err_mm':>9} {'width':>8} {'can_dz_cm':>10} {'ticks':>6}  verdict")
    print("-" * 74)
    for off_cm in OFFSETS_CM:
        obs = sim.reset(init_state_idx=0)
        objs = sim.pack(sim._obs)["objects"]
        pos = np.asarray(objs[SOUP], dtype=float)
        z_target = float(pos[2]) + off_cm / 100.0

        out = adapter.descend_to((float(pos[0]), float(pos[1])), z_target)
        eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
        adapter.set_gripper("close")
        width = float(adapter._gripper_now())
        adapter.move("+z", 8.0)
        can_after = float(sim.pack(sim._obs)["objects"][SOUP][2])
        dz_cm = (can_after - pos[2]) * 100.0

        if width < 0.006:
            verdict = "空夹"
        elif dz_cm > 5.0:
            verdict = "夹住并抬起 ✅"
        else:
            verdict = "夹住但没抬起（压住/推走）"
        print(
            f"{off_cm:>6.1f}c {eef[2]:>8.4f} {abs(eef[2] - z_target) * 1000:>9.1f} "
            f"{width:>8.4f} {dz_cm:>10.1f} {out.steps:>6}  {verdict}"
        )
        print(f"        note: {out.note[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
