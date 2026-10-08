"""What else is on the table, and does it sit in the carry lane?

Why this exists: v9, v10 and v11 all dropped the can at the SAME place
(-0.024, 0.144, z: 0.126 -> 0.047). With temperature=0 the three runs replay an
identical action sequence, so this is a deterministic failure, not bad luck --
and it happens ~11 cm short of the basket, i.e. the can is not hitting the
basket.

The can is carried at centre z ~= 0.126, i.e. its BOTTOM is at ~0.072 (the can
is ~10.7 cm tall). Anything on the table taller than ~7 cm, sitting between the
pick point (y=-0.24) and the basket (y=+0.25), would be swept by the payload.

Self-checks:
  1. the two objects we know about must be present (can, basket) at the
     documented coordinates -- else we are measuring a different scene;
  2. the can's carried bottom height must be computed, not assumed.

Run: bash scripts/simenv.sh scripts/probe_scene_objects.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

CAN_START = np.array([-0.1191, -0.2398, 0.0384])
BASKET = np.array([0.0148, 0.2521, -0.0046])
DROP_XY = np.array([-0.0236, 0.1439])

CAN_HALF_HEIGHT = 0.0535  # measured: can centre at 0.0384 when resting on table
CARRY_Z = 0.1255  # v9/v10/v11 t16: can centre while carried


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    sim.reset(init_state_idx=0)
    objs = sim.pack(sim._obs)["objects"]

    print(f"table objects ({len(objs)}):")
    print(f"{'name':<28} {'x':>8} {'y':>8} {'z':>8}   {'top z':>8}")
    print("-" * 68)
    for name in sorted(objs):
        p = np.asarray(objs[name], dtype=float)
        # we only know the can's half-height; for others just show centre
        top = "?" if "soup" not in name else f"{p[2] + CAN_HALF_HEIGHT:.4f}"
        print(f"{name:<28} {p[0]:>+8.4f} {p[1]:>+8.4f} {p[2]:>+8.4f}   {top:>8}")

    # ---- the carry lane -----------------------------------------------------
    carry_bottom = CARRY_Z - CAN_HALF_HEIGHT
    print()
    print(f"can carried at centre z={CARRY_Z:.4f} -> its BOTTOM is at z={carry_bottom:.4f}")
    print(f"basket rim is z~=0.115 (measured earlier), pick point y={CAN_START[1]:+.3f}, "
          f"basket y={BASKET[1]:+.3f}")
    print()
    print("objects whose (x,y) lies inside the straight carry lane AND whose centre z")
    print("is above the can's carried bottom (i.e. candidates to be swept):")
    hits = []
    for name, pos in objs.items():
        if "soup" in name:
            continue
        p = np.asarray(pos, dtype=float)
        # lane: between pick point and basket, with 8 cm lateral slack
        if not (min(CAN_START[1], BASKET[1]) - 0.08 <= p[1] <= max(CAN_START[1], BASKET[1]) + 0.08):
            continue
        if abs(p[0] - DROP_XY[0]) > 0.20:
            continue
        if p[2] > carry_bottom:
            hits.append((name, p))
            print(f"  ** {name:<26} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    if not hits:
        print("  (none)")

    print()
    failures = 0
    # self-check 1: the scene must contain the can and the basket where we think
    can = [k for k in objs if "soup" in k]
    bas = [k for k in objs if "basket" in k]
    ok1 = bool(can) and bool(bas)
    print(f"[{'PASS' if ok1 else 'FAIL'}] scene has can + basket: can={can} basket={bas}")
    failures += 0 if ok1 else 1
    if can:
        got = np.asarray(objs[can[0]], dtype=float)
        d = float(np.linalg.norm(got - CAN_START))
        ok2 = d < 0.005
        print(f"[{'PASS' if ok2 else 'FAIL'}] can starts where v9/v10/v11 saw it: "
              f"delta={d * 1000:.1f} mm")
        failures += 0 if ok2 else 1

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
