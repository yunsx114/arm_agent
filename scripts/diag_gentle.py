"""Diagnose why gentle mode did not engage during the carry sweep."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter, GENTLE_STEP_M  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    obs = sim.reset(init_state_idx=0)
    adapter = ActionAdapter(sim)
    key = [k for k in obs["objects"] if "soup" in k][0]
    cx, cy, cz = obs["objects"][key]

    adapter.descend_to((cx, cy), cz + 0.012)
    r = adapter.set_gripper("close")
    print(f"close note : {r.note}")
    print(f"carrying?  : {adapter._carrying()}   width={adapter._gripper_now():.4f}")
    opening = adapter._opening_axis()
    print(f"opening ax : {None if opening is None else [round(float(v), 3) for v in opening]}")
    if opening is not None:
        import numpy as np

        for d in ("+x", "-x", "+y", "-y", "+z", "-z"):
            v = np.asarray({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[d[1]]) * (1 if d[0] == "+" else -1)
            print(f"   align({d}) = {abs(float(v @ opening)):.2f}")

    m = adapter.move("+y", 2.0)
    print(f"move +y    : ticks={m.steps} note={m.note!r}")
    m2 = adapter.move("+x", 2.0)
    print(f"move +x    : ticks={m2.steps} note={m2.note!r}")
    print(f"GENTLE_STEP_M = {GENTLE_STEP_M}")

    # Full L carry with per-move notes: does gentle stay engaged for 5 cm steps?
    print("\n--- full L carry (+x15 then +y15) ---")
    obs = sim.reset(init_state_idx=0)
    key = [k for k in obs["objects"] if "soup" in k][0]
    cx, cy, cz = obs["objects"][key]
    adapter.descend_to((cx, cy), cz + 0.012)
    print(f"close: {adapter.set_gripper('close').note}")
    for direction, dist in [
        ("+z", 5), ("+z", 5),
        ("+x", 5), ("+x", 5), ("+x", 5),
        ("+y", 5), ("+y", 5), ("+y", 5),
    ]:
        out = adapter.move(direction, dist)
        print(f"  move {direction} {dist}cm: ticks={out.steps:>3}  note={out.note!r}")
    import numpy as np

    eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
    can = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
    print(f"  end |eef-can| = {np.linalg.norm(can - eef) * 1000:.1f} mm  "
          f"can_z = {can[2]:.4f} ({'HELD' if can[2] > 0.08 else 'DROPPED'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
