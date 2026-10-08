"""Where can the gripper actually reach the table?

Why this exists: v10 (t28 -> t94, 71 turns) failed to grasp a can sitting at
(0.008, 0.143, z=0.028). `descend_to(can, 1.2)` reported **xy err 1.8 mm** but
**z err 65.9 mm** -- the pads stopped at z=0.106 and `close` shut on air, over
and over. Nothing in the tool results said "this object is out of reach", so the
model just retried.

But v9 grasped the SAME can fine (descended to z=0.055) when it lay at
(-0.117, -0.242). So reachability depends on WHERE the object is -- which means
(a) a dropped object can land somewhere un-graspable, and (b) any prompt that
tells the model to "push the object away from the basket" can push it INTO such
a zone. Both need a map, not a guess.

Method: park the gripper above (x, y), command a long -z, and report the LOWEST z
reached. Self-checks:
  1. the initial gripper height must be the documented hover (~0.25);
  2. a point known to work in v9 (-0.119, -0.240) must reach below the can top;
  3. the point that failed in v10 (0.008, 0.143) must reproduce the ~0.10 floor.

Run: bash scripts/simenv.sh scripts/probe_reach_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

# x: can start (-0.119), v10 drop (0.008), basket (0.015), right side
# y: can start (-0.240), v10 drop (0.143), basket (0.252)
XS = [-0.12, -0.04, 0.0, 0.04, 0.08]
YS = [-0.24, -0.12, 0.0, 0.14, 0.25]
CAN_TOP_Z = 0.028 + 0.054  # can centre z + half its height (~10.7 cm tall)


def goto_xy(adapter: ActionAdapter, x: float, y: float) -> None:
    """Single-axis closed-loop moves to (x, y) while parked high."""
    for axis, want in (("x", x), ("y", y)):
        cur = adapter._eef_now()
        idx = 0 if axis == "x" else 1
        d = want - float(cur[idx])
        if abs(d) < 0.002:
            continue
        direction = ("+" if d > 0 else "-") + axis
        # move() caps at 15 cm per call; loop until close enough.
        for _ in range(6):
            cur = adapter._eef_now()
            d = want - float(cur[idx])
            if abs(d) < 0.004:
                break
            adapter.move(direction, min(abs(d) * 100.0, 15.0))


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)
    sim.reset(init_state_idx=0)

    start = adapter._eef_now()
    print(f"gripper hover at reset: ({start[0]:+.3f}, {start[1]:+.3f}, {start[2]:+.3f})")
    checks: list[tuple[str, bool, str]] = []
    checks.append(("hover z is ~0.25", abs(float(start[2]) - 0.25) < 0.05, f"z={start[2]:.3f}"))

    print()
    print(f"{'x':>7} {'y':>7} {'lowest_z':>10}  {'reaches can?':>13}")
    print("-" * 44)
    grid: dict[tuple[float, float], float] = {}
    for x in XS:
        for y in YS:
            sim.reset(init_state_idx=0)
            adapter.move("+z", 15)  # guarantee clearance before translating
            goto_xy(adapter, x, y)
            # move() clamps a single call to 15 cm, so "go down 25 cm" is
            # silently a 15 cm move. The probe's own self-check caught this: it
            # lifted +15 cm first, then "descended 25", i.e. ended up right back
            # where it started, and every cell read ~0.245.
            for _ in range(3):
                adapter.move("-z", 15)
            low = float(adapter._eef_now()[2])
            grid[(x, y)] = low
            ok = low < CAN_TOP_Z
            print(f"{x:>+7.3f} {y:>+7.3f} {low:>10.4f}  {'YES' if ok else 'no':>13}")

    # ---- self-checks against the two observed runs --------------------------
    low_v9 = grid[(-0.12, -0.24)]
    checks.append(("v9 start point is reachable", low_v9 < 0.06,
                   f"lowest={low_v9:.4f} (v9 descended to 0.055)"))
    # nearest sampled cell to the v10 drop (0.008, 0.143)
    low_v10 = grid[(0.0, 0.14)]
    checks.append(("v10 drop point reproduces the ~0.10 floor",
                   0.09 < low_v10 < 0.115, f"lowest={low_v10:.4f} (v10 stalled at 0.106)"))

    print()
    failures = 0
    for name, ok, detail in checks:
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    reachable = sum(1 for v in grid.values() if v < CAN_TOP_Z)
    print(f"\nreachable cells: {reachable}/{len(grid)} sampled")
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
