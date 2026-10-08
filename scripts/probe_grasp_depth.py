"""Calibrate the GRASP DEPTH: sweep z_offset_cm and measure grip quality.

WHY (evidence chain so far)
---------------------------
  * friction is NOT the bottleneck: pad slide friction is already 2.0 with a
    +/-20 N clamp (load ~1-2 N), and ablating it 2x/4x/10x/40x gives
    DROPPED / DROPPED / HELD / DROPPED -- not monotonic, i.e. luck.
  * carrying ALONG the gripper's opening axis sheds the can -> fixed in the
    adapter by auto-slowing that direction (GENTLE_STEP_M).
  * the remaining suspect is the CONTACT GEOMETRY: how deep and how centred the
    pads sit on the can. A grasp that is shallow or off-centre slides even in
    the safe direction (observed: can-eef 26 mm drifted 6 mm during +x; the
    13 mm grasp did not).

The grasp height is hard-coded as `obj_z + 1.2 cm`. This probe sweeps the
offset and, for each value, reports
    close_w      pad width after close (how much the can spreads the pads)
    |eef-can|    contact-centring proxy, split into xy and z
    carry        does it survive the +x15 then +y15 path?

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_grasp_depth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

OFFSETS_CM = [0.3, 0.6, 1.2, 2.0, 3.0]
L_CARRY = [
    ("+z", 5), ("+z", 5),
    ("+x", 5), ("+x", 5), ("+x", 5),
    ("+y", 5), ("+y", 5), ("+y", 5),
]


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    print(f"{'z_off':>6} {'close_w':>8} {'dxy':>6} {'dz':>6} {'|eef-can|':>10} "
          f"{'carry':>8}  verdict")
    print("-" * 62)

    for off_cm in OFFSETS_CM:
        obs = sim.reset(init_state_idx=0)
        key = [k for k in obs["objects"] if "soup" in k][0]
        cx, cy, cz = obs["objects"][key]

        adapter.descend_to((cx, cy), cz + off_cm / 100.0)

        eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
        can = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
        delta = can - eef
        dxy = float(np.linalg.norm(delta[:2])) * 1000.0
        dz = float(delta[2]) * 1000.0
        d3 = float(np.linalg.norm(delta)) * 1000.0

        w = float(adapter.set_gripper("close").note.split("width ")[1].split()[0])

        for direction, dist in L_CARRY:
            adapter.move(direction, dist)

        can2 = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
        eef2 = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
        ride = float(np.linalg.norm(can2 - eef2)) * 1000.0
        held = bool(can2[2] > 0.08)
        print(
            f"{off_cm:>6.1f} {w:>8.4f} {dxy:>6.1f} {dz:>6.1f} {d3:>10.1f} "
            f"{ride:>8.1f}  {'HELD' if held else 'DROPPED'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
