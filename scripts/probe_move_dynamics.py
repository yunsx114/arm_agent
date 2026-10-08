"""WHY does `move` shake the can off? Per-tick trace of a HELD vs a DROPPED carry.

The user's framing is right: a carry_to primitive does not fix this (the can can
drop just as easily above the basket, and then it may miss the basket too). The
real question is what the MOVE does to the payload, and whether it can be made
smooth -- no oscillation, no large acceleration steps.

METHOD (measure, do not guess)
------------------------------
Monkeypatch `sim.step` to record, on EVERY control tick:
    eef position, can position, and the commanded 7-D action.
Then run the two carries whose outcomes we already know differ:
    +x15      (3 move calls)  -> HELD
    +x15+y15  (6 move calls)  -> DROPPED
and report, per tick:
    |eef - can|          (the rider distance; a jump = the can slipped)
    eef displacement     (mm per tick)
    eef delta-v          (mm per tick^2 -> the acceleration step)
so the slip can be matched to the dynamics that caused it.

Run:  cd arm_agent && bash scripts/simenv.sh scripts/probe_move_dynamics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

CARRY = {
    "+x15 (HELD)": [("+z", 5), ("+z", 5), ("+x", 5), ("+x", 5), ("+x", 5)],
    "+x15+y15 (DROPPED)": [
        ("+z", 5), ("+z", 5),
        ("+x", 5), ("+x", 5), ("+x", 5),
        ("+y", 5), ("+y", 5), ("+y", 5),
    ],
}


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(x,y,z,w) quaternion -> 3x3 rotation matrix (columns = body axes in world)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    trace: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]] = []
    orig_step = sim.step
    call_no = 0

    def rec_step(action):  # noqa: ANN001
        nonlocal call_no
        out = orig_step(action)
        eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
        can = np.asarray(sim._obs["alphabet_soup_1_pos"], dtype=float)
        quat = np.asarray(sim._obs["robot0_eef_quat"], dtype=float)
        trace.append((eef.copy(), can.copy(), np.asarray(action, dtype=float).copy(), call_no, quat.copy()))
        return out

    sim.step = rec_step  # type: ignore[assignment]

    for label, actions in CARRY.items():
        obs = sim.reset(init_state_idx=0)
        key = [k for k in obs["objects"] if "soup" in k][0]
        cx, cy, cz = obs["objects"][key]
        adapter.descend_to((cx, cy), cz + 0.012)
        w = float(adapter.set_gripper("close").note.split("width ")[1].split()[0])

        trace.clear()
        call_no = 0
        for i, (direction, dist) in enumerate(actions):
            call_no = i  # tag every tick of this move() call with its index
            adapter.move(direction, dist)

        eefs = np.array([t[0] for t in trace])
        cans = np.array([t[1] for t in trace])
        quats = np.array([t[4] for t in trace])
        rel = np.linalg.norm(cans - eefs, axis=1) * 1000.0  # mm
        step_mm = np.linalg.norm(np.diff(eefs, axis=0), axis=1) * 1000.0  # mm/tick
        dstep = np.abs(np.diff(step_mm))  # mm/tick^2

        # The gripper's opening axis in WORLD coordinates (robosuite Panda pads
        # travel along the EEF body y axis). A payload is pushed OUT most easily
        # when the carry direction lies along this axis.
        R = quat_to_mat(quats[len(quats) // 2])
        open_axis = R[:, 1]
        open_axis = open_axis / (np.linalg.norm(open_axis) + 1e-12)
        carry = np.asarray(eefs[-1] - eefs[0], dtype=float)
        carry = carry / (np.linalg.norm(carry) + 1e-12)
        print(f"\n=== {label}   close_w={w:.4f}   ticks={len(trace)} ===")
        print(f"  gripper opening axis (world): ({open_axis[0]:+.2f}, {open_axis[1]:+.2f}, {open_axis[2]:+.2f})")
        print(f"  net carry direction         : ({carry[0]:+.2f}, {carry[1]:+.2f}, {carry[2]:+.2f})")
        print(f"  |cos(angle)| carry vs opening axis = {abs(float(open_axis @ carry)):.2f}")
        print(f"  |eef-can|: start {rel[0]:.1f}mm  end {rel[-1]:.1f}mm  max {rel.max():.1f}mm")
        if len(rel) > 1:
            jumps = np.diff(rel)
            k = int(np.argmax(np.abs(jumps)))
            print(f"  biggest slip: tick {k + 1}  d_rel={jumps[k]:+.2f}mm")
            print(f"  eef travel: max {step_mm.max():.1f} mm/tick, "
                  f"max |delta-v| {dstep.max():.1f} mm/tick^2")
            lo, hi = max(0, k - 4), min(len(trace), k + 5)
            print("  ticks around the slip (call, eef_move, rel, action):")
            for i in range(lo, hi):
                e, c, a, cn, _q = trace[i]
                mv = step_mm[i - 1] if i > 0 else 0.0
                dv = dstep[i - 1] if i > 0 else 0.0
                d3 = (e - trace[i - 1][0]) * 1000.0 if i > 0 else np.zeros(3)
                mark = "  <-- SLIP" if i == k + 1 else ""
                print(
                    f"    tick{i:>3} move#{cn} step={mv:5.1f}mm dv={dv:5.1f} "
                    f"dx={d3[0]:+5.1f} dy={d3[1]:+5.1f} dz={d3[2]:+5.1f} "
                    f"rel={rel[i]:6.1f}mm{mark}"
                )

    sim.step = orig_step  # type: ignore[assignment]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
