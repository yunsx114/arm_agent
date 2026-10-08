"""Tick-level descent probe: WHY does a -z move introduce XY drift?

The dry run showed the policy correcting ~0.7 cm of x error after EVERY 2 cm
descent step, which makes the descent oscillate instead of converging. This
probe logs the EEF position on every internal tick of a controlled descent so
the cause is measured, not guessed.

It runs the adapter's own `move()` servo and records the trace by monkeypatching
`sim.step`. Two descents are compared:

  A) with the adapter's 3-D servo to a fixed target (x0, y0, z0 - 20 cm)
  B) a pure -z command with NO xy correction (raw drift baseline)

Output: per-tick table + a summary of where XY error enters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

SOUP = (-0.119, -0.240)
HOVER_Z = 0.25


def go_xy_then_descend(sim: LiberoSim, adapter: ActionAdapter, soup_xy: tuple[float, float]) -> None:
    """Move above the can with the adapter servo, then descend 20 cm."""
    for _ in range(60):
        x, y, z = sim._obs["robot0_eef_pos"]
        dx, dy = soup_xy[0] - x, soup_xy[1] - y
        if abs(dx) > 0.002:
            adapter.move("+x" if dx > 0 else "-x", min(5.0, abs(dx) * 100))
        elif abs(dy) > 0.002:
            adapter.move("+y" if dy > 0 else "-y", min(5.0, abs(dy) * 100))
        elif abs(z - HOVER_Z) > 0.005:
            adapter.move("+z" if z < HOVER_Z else "-z", min(5.0, abs(z - HOVER_Z) * 100))
        else:
            return
    raise RuntimeError("go_xy_then_descend did not converge")


def trace_descent(sim: LiberoSim, adapter: ActionAdapter, distance_cm: float, label: str) -> None:
    """Run adapter.move('-z', distance_cm) and record eef on every tick."""
    trace: list[tuple[float, float, float]] = []

    original_step = sim.step

    def recording_step(action):
        obs = original_step(action)
        eef = obs["eef_pos"]
        trace.append((eef[0], eef[1], eef[2]))
        return obs

    sim.step = recording_step  # type: ignore[method-assign]
    try:
        adapter.move("-z", distance_cm)
    finally:
        sim.step = original_step  # type: ignore[method-assign]

    x0, y0, z0 = trace[0] if trace else (0, 0, 0)
    print(f"\n=== {label}: {len(trace)} ticks ===")
    print(f"{'tick':>4s} {'x':>9s} {'y':>9s} {'z':>9s} {'dx_mm':>7s} {'dy_mm':>7s} {'dz_mm':>7s}")
    for i, (x, y, z) in enumerate(trace):
        print(
            f"{i:4d} {x:9.4f} {y:9.4f} {z:9.4f} "
            f"{(x - x0) * 1000:7.1f} {(y - y0) * 1000:7.1f} {(z - z0) * 1000:7.1f}"
        )
    if trace:
        x1, y1, z1 = trace[-1]
        print(
            f"NET: dx={(x1 - x0) * 1000:+.1f}mm dy={(y1 - y0) * 1000:+.1f}mm "
            f"dz={(z1 - z0) * 1000:+.1f}mm"
        )
        xmax = max(abs(x - x0) for x, _, _ in trace) * 1000
        ymax = max(abs(y - y0) for _, y, _ in trace) * 1000
        print(f"MAX |xy drift| during move: x={xmax:.1f}mm y={ymax:.1f}mm")


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    print("### A) servo descent (adapter.move -z 20cm) from hover")
    sim.reset(init_state_idx=0)
    go_xy_then_descend(sim, adapter, SOUP)
    x, y, z = sim._obs["robot0_eef_pos"]
    print(f"hover: ({x:.4f}, {y:.4f}, {z:.4f})  err_vs_soup=({(x - SOUP[0]) * 1000:+.1f},{(y - SOUP[1]) * 1000:+.1f})mm")
    trace_descent(sim, adapter, 20.0, "A servo descent")

    print("\n\n### B) pure -z, no xy servo (raw drift baseline), 10 ticks of 2cm")
    sim.reset(init_state_idx=0)
    go_xy_then_descend(sim, adapter, SOUP)
    x0, y0, z0 = sim._obs["robot0_eef_pos"]
    for i in range(10):
        action = np.zeros(7)
        action[2] = -2.0 / 5.0  # 2 cm down per tick, gripper hold
        sim.step(action)
        x, y, z = sim._obs["robot0_eef_pos"]
        print(f"tick {i}: ({x:.4f}, {y:.4f}, {z:.4f})  d_mm=({(x - x0) * 1000:+.1f},{(y - y0) * 1000:+.1f},{(z - z0) * 1000:+.1f})")

    print("\n\n### C) FINE descent in 5 mm chunks: where does contact happen?")
    print("prints eef xyz, can xyz, finger gap after EVERY tick")
    sim.reset(init_state_idx=0)
    go_xy_then_descend(sim, adapter, SOUP)
    x, y, z = sim._obs["robot0_eef_pos"]
    can = sim._obs["alphabet_soup_1_pos"]
    print(f"start: eef=({x:+.4f},{y:+.4f},{z:+.4f}) can=({can[0]:+.4f},{can[1]:+.4f},{can[2]:+.4f})")
    # use the adapter's own servo (so the dry run's exact mechanics are used),
    # 5 mm per call, and let the trace print the mesh state each tick
    original_step = sim.step
    tick_no = [0]

    def recording_step(action):
        obs = original_step(action)
        tick_no[0] += 1
        if tick_no[0] % 2 == 0:  # every other tick keeps the log readable
            # NB: with cameras disabled the packed obs has no object keys, so
            # only the EEF (always present) is logged here.
            e = obs["robot0_eef_pos"]
            g = float(np.abs(np.asarray(obs["robot0_gripper_qpos"])).mean())
            print(f"  t{tick_no[0]:3d} eef=({e[0]:.4f},{e[1]:.4f},{e[2]:.4f}) grip={g:.4f}")
        return obs

    sim.step = recording_step  # type: ignore[method-assign]
    try:
        for i in range(30):
            z = sim._obs["robot0_eef_pos"][2]
            if z <= 0.03:
                break
            adapter.move("-z", 0.5)
    finally:
        sim.step = original_step  # type: ignore[method-assign]

    e = sim._obs["robot0_eef_pos"]
    c = sim._obs["alphabet_soup_1_pos"]
    print(f"end: eef_z={e[2]:.4f} can=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})")

    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
