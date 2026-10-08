"""M1 grasp calibration: sweep the descent height and find where the gripper
actually captures the alphabet soup can (ideal XY alignment, GT targets).

The dry run showed the can being PUSHED 7 cm instead of grasped, so the
descend height and alignment tolerance are the variables to measure. This probe
removes alignment from the equation (greedy closed loop against GT coordinates,
2 mm tolerance) so that only the height remains:

    for z in {0.02, 0.035, 0.05, 0.065, 0.08}:
        reset -> go above can (GT) -> descend to z -> close -> lift +8 cm
        success = can z rises by >5 cm and close width is non-zero (something
                  between the fingers)

Also reports the close-time finger width, the diagnostic that separates
"grasped" (~can radius) from "closed on air" (~0) or "pushed the can".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter
from arm_agent.sim.libero_env import LiberoSim

SOUP = (-0.119, -0.240)  # GT for init state 0 (alphabet_soup_1_pos)
HOVER_Z = 0.25
XY_TOL = 0.002
# Z tolerance is looser than XY: against resistance (can top, table) the OSC
# settles with a 1-3 mm standing error that never satisfies a 2 mm window.
Z_TOL = 0.005
LIFT_CM = 8.0


def go_to(adapter: ActionAdapter, sim: LiberoSim, target_xy: tuple[float, float],
          target_z: float) -> None:
    """Greedy closed loop toward (x, y, z) with adapter-level moves.

    Includes a stuck detector: if six iterations move the EEF less than 1 mm in
    total while still outside tolerance, the target is unreachable (usually the
    arm pressing against an obstacle) and we bail instead of burning ticks.
    """
    last = None
    still = 0
    for _ in range(80):
        x, y, z = sim._obs["robot0_eef_pos"]
        if last is not None:
            if abs(x - last[0]) + abs(y - last[1]) + abs(z - last[2]) < 0.001:
                still += 1
                if still >= 6:
                    raise RuntimeError(f"go_to stuck at ({x:.3f}, {y:.3f}, {z:.3f})")
            else:
                still = 0
        last = (x, y, z)
        dx, dy, dz = target_xy[0] - x, target_xy[1] - y, target_z - z
        # Descend last so we never drag across the table into the can.
        if abs(dx) > XY_TOL:
            adapter.move("+x" if dx > 0 else "-x", min(5.0, abs(dx) * 100))
        elif abs(dy) > XY_TOL:
            adapter.move("+y" if dy > 0 else "-y", min(5.0, abs(dy) * 100))
        elif abs(dz) > Z_TOL:
            adapter.move("+z" if dz > 0 else "-z", min(5.0, abs(dz) * 100))
        else:
            return
    raise RuntimeError("go_to did not converge")


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    print(f"{'z_target':>8s} {'xy_err_cm':>9s} {'close_width':>11s} "
          f"{'can_dz_cm':>9s} {'eef_xy_err_cm':>13s} verdict")
    results = []
    for z_target in (0.035, 0.05, 0.065, 0.08, 0.095):
        sim.reset(init_state_idx=0)
        can0 = sim._obs["alphabet_soup_1_pos"]
        go_to(adapter, sim, SOUP, HOVER_Z)
        x, y, z = sim._obs["robot0_eef_pos"]
        xy_err = ((x - SOUP[0]) ** 2 + (y - SOUP[1]) ** 2) ** 0.5
        go_to(adapter, sim, SOUP, z_target)
        x, y, z = sim._obs["robot0_eef_pos"]
        eef_xy_err = ((x - SOUP[0]) ** 2 + (y - SOUP[1]) ** 2) ** 0.5

        adapter.set_gripper("close")
        width = adapter._gripper_now()
        adapter.move("+z", LIFT_CM)

        can1 = sim._obs["alphabet_soup_1_pos"]
        dz_cm = (can1[2] - can0[2]) * 100
        pushed = ((can1[0] - can0[0]) ** 2 + (can1[1] - can0[1]) ** 2) ** 0.5
        if dz_cm > 5:
            verdict = "GRASPED"
        elif pushed > 0.02:
            verdict = f"pushed {pushed * 100:.0f}cm"
        else:
            verdict = "missed (air)"
        results.append((z_target, dz_cm, verdict))
        print(f"{z_target:8.3f} {xy_err * 100:9.2f} {width:11.4f} "
              f"{dz_cm:9.1f} {eef_xy_err * 100:13.2f} {verdict}", flush=True)

    sim.close()
    best = max(results, key=lambda r: r[1])
    print(f"\nbest: z={best[0]} can_dz={best[1]:.1f}cm ({best[2]})")
    return 0 if best[1] > 5 else 1


if __name__ == "__main__":
    sys.exit(main())
