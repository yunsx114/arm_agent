"""Can the friction coefficient be tuned? Read the ACTUAL parameters first.

The user asked whether friction is adjustable. Technically yes: MuJoCo keeps
per-geom friction in `model.geom_friction = [slide, spin, roll]`, and robosuite
exposes the raw model, so it can be changed at runtime.

IMPORTANT CAVEAT: changing it changes the PHYSICS OF THE BENCHMARK. Results then
stop being comparable to LIBERO's published numbers, and "it works now" only
means "it works under my own friction". So before touching it, measure what the
sim currently uses -- and check whether the real deficit is friction at all, or
the gripper's FORCE budget (a weak clamp looks exactly like low friction).

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_grasp_physics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    sim.reset(init_state_idx=0)
    robosuite_env = sim.env.env  # robosuite wrapper inside LIBERO's env
    m = robosuite_env.sim.model

    print(f"ngeom={m.ngeom}  nu={m.nu}")

    print(f"model type: {type(m).__module__}.{type(m).__name__}")

    names: list[str] = []
    try:
        GJ = int(mujoco.mjtObj.mjOBJ_GEOM)
        names = [mujoco.mj_id2name(m, GJ, i) or f"geom{i}" for i in range(m.ngeom)]
    except Exception as exc:  # noqa: BLE001
        print(f"  [geom name lookup unavailable: {type(exc).__name__}: {exc}]")

    print("\n=== geoms named like pads / gripper / can ===")
    keys = ("soup", "finger", "pad", "gripper", "right", "left", "can")
    if names:
        for i, name in enumerate(names):
            if any(k in name for k in keys):
                f = m.geom_friction[i]
                print(f"  {i:>3} {name:46} [{f[0]:.3f}, {f[1]:.6f}, {f[2]:.6f}]")
    else:
        print("  (skipped)")

    print("\n=== friction VALUES in use (unique) ===")
    uniq, counts = np.unique(np.asarray(m.geom_friction), axis=0, return_counts=True)
    for row, cnt in zip(uniq, counts):
        print(f"  [{row[0]:.3f}, {row[1]:.6f}, {row[2]:.6f}]  x{cnt}")

    print("\n=== actuators (the clamp force budget) ===")
    try:
        AJ = int(mujoco.mjtObj.mjOBJ_ACTUATOR)
        anames = [mujoco.mj_id2name(m, AJ, i) or f"act{i}" for i in range(m.nu)]
    except Exception:  # noqa: BLE001
        anames = [f"act{i}" for i in range(m.nu)]
    for i in range(m.nu):
        name = anames[i]
        fr = m.actuator_forcerange[i]
        cr = m.actuator_ctrlrange[i]
        gp = np.asarray(m.actuator_gainprm[i])[:3]
        inf = "  <-- gripper" if ("grip" in name or "finger" in name) else ""
        print(
            f"  {name:34} ctrl[{cr[0]:+.3f},{cr[1]:+.3f}] "
            f"force[{fr[0]:+.1f},{fr[1]:+.1f}] gainprm={np.round(gp, 2)}{inf}"
        )

    # MuJoCo's own defaults, for reference.
    print("\n=== reference ===")
    print("  MuJoCo default geom_friction = [1.0, 0.005, 0.0001]")
    print("  a real aluminium can on rubber pads: slide friction ~0.4-0.6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
