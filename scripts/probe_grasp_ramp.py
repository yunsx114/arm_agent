"""Sweep GRIPPER_RAMP_EXTRA_STEPS: does the grip survive a REAL carry?

WHY this probe exists
---------------------
The current value (10) was tuned while `move()` silently did 0 ticks (the
missing `sim.step()` bug), so the arm never actually carried anything -- that
tune was made on a broken rig and its "10 is optimal, 30 slips, 100 ejects the
can" conclusion is NOT trustworthy. Now that `move` works, the hold-force
trade-off must be re-measured.

METHOD
------
For each candidate ramp value, on the SAME init state (so the can starts in the
same place), run a real carry:

    descend_to(can, 1.2cm) -> set_gripper close -> lift 10 cm -> translate 15 cm

then read the GROUND-TRUTH can position against the EEF position:
    can rider on the EEF  -> the grip held
    can back near the table -> it slipped out mid-carry

Run:  cd arm_agent && bash scripts/simenv.sh scripts/probe_grasp_ramp.py
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

CANDIDATES = [10, 30, 60, 100]
TABLE_Z = 0.04  # a can left on the table sits near this height
HELD_TOL_M = 0.04  # rider distance to the EEF that still counts as held


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    print(f"{'ramp':>5} {'close_w':>8} {'@lift_can_z':>12} {'@end_can_z':>11} "
          f"{'can-eef':>8}  verdict")
    print("-" * 66)
    rows: list[tuple[int, float, float, float, float, bool]] = []

    for ramp in CANDIDATES:
        A.GRIPPER_RAMP_EXTRA_STEPS = ramp  # set_gripper() resolves it as a global

        obs = sim.reset(init_state_idx=0)
        objs = obs["objects"]
        key = [k for k in objs if "soup" in k][0]
        cx, cy, cz = objs[key]

        # ---- grasp ----
        adapter.descend_to((cx, cy), cz + 0.012)
        out_c = adapter.set_gripper("close")
        close_w = float(out_c.note.split("width ")[1].split()[0])

        # ---- carry: lift 10 cm (2x5, move clamps at 5 cm), then 15 cm across ----
        adapter.move("+z", 5.0)
        adapter.move("+z", 5.0)
        lift_can = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
        for _ in range(3):
            adapter.move("+x", 5.0)

        eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
        can = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
        dist = float(np.linalg.norm(can - eef))
        held = bool(can[2] > TABLE_Z + 0.04 and dist < HELD_TOL_M)
        rows.append((ramp, close_w, float(lift_can[2]), float(can[2]), dist, held))
        print(
            f"{ramp:>5} {close_w:>8.4f} {lift_can[2]:>12.4f} {can[2]:>11.4f} "
            f"{dist:>8.4f}  {'HELD' if held else 'DROPPED'}"
        )

    print()
    ok = [r for r in rows if r[5]]
    if ok:
        best = min(ok, key=lambda r: r[4])
        print(
            f"=> {len(ok)}/{len(rows)} survived the carry. Tightest hold: ramp={best[0]} "
            f"(can-eef {best[4] * 100:.1f} cm, close width {best[1]:.4f})"
        )
    else:
        print("=> NO value survived the carry (the grip is the bottleneck, not the ramp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
