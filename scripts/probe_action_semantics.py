"""M1 action-semantics probe: verify move/rotate/gripper against MEASURED geometry.

Run before trusting the adapter in a live episode (measurement discipline:
"direction conclusions must be read from geometry, not derived from formulas"):

    cd arm_agent && bash scripts/simenv.sh scripts/probe_action_semantics.py

Assertions (with tolerances that reflect OSC tracking, not ideals):
  move +x 5cm   -> eef x increases by roughly +0.025..0.06 m
  move -y 5cm   -> eef y decreases
  move +z 5cm   -> eef z increases
  rotate yaw +30 -> eef quat rotates about world z by roughly +30 deg
  set_gripper close -> finger width collapses toward 0
  set_gripper open  -> finger width back to ~0.0208
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from arm_agent.sim.action_adapter import ActionAdapter
from arm_agent.sim.libero_env import LiberoSim


def axis_angle_between(q0: np.ndarray, q1: np.ndarray) -> tuple[np.ndarray, float]:
    """Relative rotation q1 * q0^-1 as (axis, angle in deg); quats are (x,y,z,w)."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    # conjugate of q0
    x0c, y0c, z0c, w0c = -x0, -y0, -z0, w0
    # q_rel = q1 * q0_conj
    w = w1 * w0c - x1 * x0c - y1 * y0c - z1 * z0c
    x = w1 * x0c + x1 * w0c + y1 * z0c - z1 * y0c
    y = w1 * y0c - x1 * z0c + y1 * w0c + z1 * x0c
    z = w1 * z0c + x1 * y0c - y1 * x0c + z1 * w0c
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return np.zeros(3), 0.0
    angle = 2 * math.degrees(math.acos(max(-1.0, min(1.0, w))))
    return np.array([x, y, z]) / n, angle


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    sim.reset(init_state_idx=0)
    adapter = ActionAdapter(sim)

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print("=== move semantics ===")
    for direction, axis_index, sign in (("+x", 0, +1), ("-y", 1, -1), ("+z", 2, +1)):
        before = sim._obs["robot0_eef_pos"]
        out = adapter.move(direction, 5.0)
        delta = np.asarray(out.measured_delta_m)
        moved = delta[axis_index] * sign
        # Closed loop: within the 5 mm arrival tolerance of the 50 mm request.
        check(
            f"move {direction} 5cm",
            0.040 < moved < 0.060,
            f"measured delta {delta.tolist()} ({out.steps} ticks)",
        )
        # cross-axis leakage should be small
        other = [abs(delta[i]) for i in range(3) if i != axis_index]
        check(
            f"move {direction} cross-axis leakage",
            max(other) < 0.02,
            f"off-axis {max(other):.4f} m",
        )

    print("\n=== rotate semantics ===")
    q0 = np.asarray(sim._obs["robot0_eef_quat"]).copy()
    out = adapter.rotate("yaw", 30.0)
    q1 = np.asarray(sim._obs["robot0_eef_quat"])
    axis, angle = axis_angle_between(q0, q1)
    check(
        "rotate yaw +30 about world z",
        abs(axis[2]) > 0.95 and abs(angle - 30) < 8,
        f"axis {np.round(axis, 3).tolist()} angle {angle:.1f} deg "
        f"({out.steps} ticks, note: {out.note})",
    )

    print("\n=== gripper semantics ===")
    w_open_before = float(np.abs(np.asarray(sim._obs["robot0_gripper_qpos"])).mean())
    out = adapter.set_gripper("close")
    w_closed = float(np.abs(np.asarray(sim._obs["robot0_gripper_qpos"])).mean())
    check(
        "close collapses finger width",
        w_closed < 0.005 < w_open_before,
        f"width {w_open_before:.4f} -> {w_closed:.4f} ({out.steps} steps)",
    )
    out = adapter.set_gripper("open")
    w_reopen = float(np.abs(np.asarray(sim._obs["robot0_gripper_qpos"])).mean())
    check(
        "open restores finger width",
        w_reopen > 0.015,
        f"width {w_closed:.4f} -> {w_reopen:.4f} ({out.steps} steps)",
    )

    # Hold semantics: move with gripper channel 0 must NOT change the gripper.
    adapter.set_gripper("open")
    out = adapter.move("+x", 2.5)
    w_after_move = float(np.abs(np.asarray(sim._obs["robot0_gripper_qpos"])).mean())
    check(
        "move holds gripper state",
        abs(w_after_move - w_reopen) < 0.004,
        f"width {w_reopen:.4f} -> {w_after_move:.4f} after move",
    )

    sim.close()
    print(f"\n=== {len(failures)} failure(s): {failures} ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
