"""Is the TURN the problem? Carry diagonally in ONE move (no direction change).

MEASURED (probe_carry_order.py, 5 repeats each):
    +z only (lift, no lateral) : 5/5 HELD
    +x15 only                  : 4/5 HELD
    +y15 only                  : 5/5 HELD    <-- lateral, and ALONG the opening axis
    +x15 then +y15 (L path)    : 1/5 HELD    <-- same endpoint; only a turn added

So neither the axis nor lateral motion is fatal by itself -- the L-TURN is. The
payload survives any single straight leg, and is shed when the motion changes
direction mid-carry.

Hypothesis: the adapter's `move` is SINGLE-AXIS, so an L path is really a
sequence of two straight servos with a stop-and-turn between them; during the
turn the payload keeps its momentum while the pads change direction.

Test: reach the SAME endpoint (15, 15) in ONE straight diagonal, both axes
driven on every tick, so there is no turn at all.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_diagonal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

REPEATS = 5
CAP_M = 0.01  # per-tick displacement, same as the adapter's gentle mode

# Each config is a list of (dx_cm, dy_cm, dz_cm) legs. A leg is driven as ONE
# straight servo (all axes active every tick), so a 2-leg config has exactly one
# turn.
CONFIGS = {
    "L: lift,x15,y15": [(0, 0, 10), (15, 0, 0), (0, 15, 0)],
    "diagonal: lift,x15y15": [(0, 0, 10), (15, 15, 0)],
}


def straight(adapter: ActionAdapter, dx_cm: float, dy_cm: float, dz_cm: float) -> int:
    """One straight-line servo in base-frame xyz. Returns ticks used."""
    target = adapter._eef_now() + np.array([dx_cm, dy_cm, dz_cm], dtype=float) / 100.0
    ticks = 0
    for _ in range(300):
        err = target - adapter._eef_now()
        if float(np.linalg.norm(err)) < 0.002:
            break
        step = err
        norm = float(np.linalg.norm(step))
        if norm > CAP_M:
            step = step / norm * CAP_M
        action = np.zeros(7)
        action[:3] = step / 0.05
        action[6] = 0.0
        adapter.sim.step(action)
        ticks += 1
    return ticks


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    for label, legs in CONFIGS.items():
        held_n = 0
        widths: list[float] = []
        rides: list[float] = []
        for _rep in range(REPEATS):
            obs = sim.reset(init_state_idx=0)
            key = [k for k in obs["objects"] if "soup" in k][0]
            cx, cy, cz = obs["objects"][key]

            adapter.descend_to((cx, cy), cz + 0.012)
            widths.append(
                float(adapter.set_gripper("close").note.split("width ")[1].split()[0])
            )
            for dx, dy, dz in legs:
                straight(adapter, dx, dy, dz)

            can = np.asarray(sim.pack(sim._obs)["objects"][key], dtype=float)
            eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
            rides.append(float(np.linalg.norm(can - eef)) * 1000.0)
            held_n += int(can[2] > 0.08)

        print(f"\n{label}:  {held_n}/{REPEATS} HELD")
        print(f"   close_w : {[round(w, 4) for w in widths]}")
        print(f"   |eef-can| after carry (mm): {[round(r) for r in rides]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
