"""Is friction the bottleneck? Ablation -- crank the pad friction and re-carry.

Measured parameters (probe_grasp_physics.py):
    pad geoms (2)   geom_friction slide = 2.0     (real rubber ~1.2, so already high)
    gripper acts    forcerange = +/-20 N,  kp = 1000
    payload         alphabet_soup ~0.1-0.2 kg -> ~1-2 N

So the friction force available at the pads (~20 N) is 10-40x the load. If the
can STILL slips with DOUBLE and QUINTUPLE pad friction, friction is not the
bottleneck, and tuning it would only hide the real cause (contact geometry /
how deep the pads sit on the can).

This also documents HOW to change it, for the record: MuJoCo keeps per-geom
friction in `model.geom_friction = [slide, spin, roll]` and robosuite exposes
the raw model, so `geom_friction[i, 0] = x` works at runtime.

WARNING: changing it changes the PHYSICS OF THE BENCHMARK. Numbers then stop
being comparable to LIBERO's published results.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_friction_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

L_CARRY = [
    ("+z", 5), ("+z", 5),
    ("+x", 5), ("+x", 5), ("+x", 5),
    ("+y", 5), ("+y", 5), ("+y", 5),
]


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)
    sim.reset(init_state_idx=0)
    rs_env = sim.env.env  # robosuite wrapper
    model = rs_env.sim.model
    data = rs_env.sim.data

    fr = np.asarray(model.geom_friction)
    orig = fr.copy()
    pads = [i for i in range(fr.shape[0]) if abs(float(fr[i, 0]) - 2.0) < 1e-6]
    print(f"pad geoms {pads}, base slide friction {[float(fr[i, 0]) for i in pads]}")
    print(f"{'pad_friction':>14} {'close_w':>8} {'can-eef':>9} {'can_z':>8}  verdict")
    print("-" * 56)

    for factor in (1.0, 2.0, 5.0, 20.0):
        fr[:, :] = orig
        for i in pads:
            fr[i, 0] = float(orig[i, 0]) * factor
        rs_env.sim.forward()  # refresh derived contact quantities

        obs = sim.reset(init_state_idx=0)
        key = [k for k in obs["objects"] if "soup" in k][0]
        cx, cy, cz = obs["objects"][key]
        adapter.descend_to((cx, cy), cz + 0.012)
        w = float(adapter.set_gripper("close").note.split("width ")[1].split()[0])
        for d, dist in L_CARRY:
            adapter.move(d, dist)

        eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
        can = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
        d_mm = float(np.linalg.norm(can - eef)) * 1000.0
        held = bool(can[2] > 0.08)
        print(
            f"{float(orig[pads[0], 0]) * factor:>14.2f} {w:>8.4f} {d_mm:>9.1f} "
            f"{can[2]:>8.4f}  {'HELD' if held else 'DROPPED'}"
        )

    fr[:, :] = orig
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
