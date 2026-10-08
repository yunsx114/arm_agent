"""Which CARRY ACTION drops the can? (the ramp is not the culprit)

The ramp sweep (probe_grasp_ramp.py) showed ramp=10 AND 30 both keep the can
through "+z 10 cm, then +x 15 cm". Yet the 80-turn episode dropped it after a
+ y 5 cm move (that turn's measured delta was 0.0189, 0.0513, 0.0136 -- i.e. it
also shoved the EEF 1.4 cm in z).

So this probe holds the ramp fixed at 10 and varies only the CARRY TRAJECTORY,
on the same init state each time:

    +x15        lift 10 cm, slide +x 15 cm      (the swept path that HELD)
    +y15        lift 10 cm, slide +y 15 cm      (the episode's last move)
    +x15+y15    lift 10 cm, +x 15 then +y 15    (the episode's actual path)
    lift15+x15  lift 15 cm first, then +x 15 cm (is height the difference?)

Run:  cd arm_agent && bash scripts/simenv.sh scripts/probe_carry_modes.py
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

RAMP = 10
REPEATS = 5
_LPATH = [
    ("+z", 5), ("+z", 5),
    ("+x", 5), ("+x", 5), ("+x", 5),
    ("+y", 5), ("+y", 5), ("+y", 5),
]
# mode -> (action list, per-tick displacement cap in metres)
# The adapter's own gentle mode already slows every move made while carrying, so
# these two should behave the SAME. Running each 5x is what separates a real
# effect from grip-to-grip luck: |eef-can| after a grasp was measured anywhere
# from 13 mm to 26 mm, and that spread alone flips the outcome.
MODES: dict[str, tuple[list[tuple[str, float]], float]] = {
    "cap 5.0 (adapter gentle)": (list(_LPATH), 0.05),
    "cap 1.0 (global slow)": (list(_LPATH), 0.01),
}
TABLE_Z = 0.04
HELD_TOL_M = 0.04


def quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    """Angle between two (x,y,z,w) quaternions, in degrees."""
    d = min(1.0, abs(float(np.dot(q0, q1))))
    return float(np.degrees(2.0 * np.arccos(d)))


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)
    A.GRIPPER_RAMP_EXTRA_STEPS = RAMP

    print(f"ramp fixed at {RAMP}; comparing carry trajectories")
    print(f"{'mode':>15} {'close_w':>8} {'after_lift':>11} {'after_carry':>12} "
          f"{'can-eef':>8} {'max_tilt':>8}  verdict")
    print("-" * 82)

    printed_objs = False
    for mode, (actions, cap) in MODES.items():
        A.MOVE_STEP_CAP_M = cap
        obs = sim.reset(init_state_idx=0)
        objs = obs["objects"]
        if not printed_objs:
            print("scene objects (is something in the carry path?):")
            for k, v in sorted(objs.items()):
                print(f"    {k:24} ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")
            printed_objs = True
        key = [k for k in objs if "soup" in k][0]
        cx, cy, cz = objs[key]

        # Grasp, retrying if the pads shut too far: a marginal pinch is what
        # gave lift15+x15+y15 its close_w=0.0135, and that run then tells us
        # nothing about the carry. Require the pad width of a REAL grasp.
        close_w = 0.0
        for _ in range(3):
            adapter.descend_to((cx, cy), cz + 0.012)
            close_w = float(adapter.set_gripper("close").note.split("width ")[1].split()[0])
            if close_w >= 0.015:
                break
            adapter.set_gripper("open")

        lift_z = float("nan")
        q_grasp = np.asarray(sim._obs["robot0_eef_quat"], dtype=float).copy()
        tilt_max = 0.0
        for i, (direction, dist) in enumerate(actions):
            adapter.move(direction, dist)
            q_now = np.asarray(sim._obs["robot0_eef_quat"], dtype=float)
            tilt_max = max(tilt_max, quat_angle_deg(q_grasp, q_now))
            if i == 1:  # after the 10 cm lift (2nd +z)
                lift_z = float(np.asarray(sim.pack(sim._obs)["objects"][key])[2])

        eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
        can = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
        dist = float(np.linalg.norm(can - eef))
        held = bool(can[2] > TABLE_Z + 0.04 and dist < HELD_TOL_M)
        print(
            f"{mode:>15} {close_w:>8.4f} {lift_z:>11.4f} {can[2]:>12.4f} "
            f"{dist:>8.4f} {tilt_max:>7.2f}d  {'HELD' if held else 'DROPPED'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
