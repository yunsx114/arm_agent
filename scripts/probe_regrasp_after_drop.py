#!/usr/bin/env python3
"""Can the can be RE-GRASPED after it is dropped, or does every descent shove it?

WHY
---
P0e (episode_000_1791510070) went: grasp at t3 -> lift -> descend onto its own
payload (t5) -> squeeze close (t6) -> drop on the next lift (t7) -> then 53 turns
of `locate / descend_to / align_xy / move` with **zero `close`**. The can was
driven from y=-0.24 to y=-0.52 while the model chased it.

From the transcript alone two stories fit:
  (a) the model never closes because it believes it still holds the can; or
  (b) every descent onto the fallen can moves it ~4-5 cm, so the `locate` right
      after the descent always shows a fresh large dy and the model keeps
      "re-aligning" instead of closing.
This probe decides between them by measuring (b) directly: whether a descent
onto a can that is ALREADY on the table displaces it.

WHAT IT MEASURES
  1. baseline -- descent onto a can that was never grasped. If THIS moves the can
     appreciably the probe's own method is broken (self-check, hard fail).
  2. the P0e replay -- grasp, lift, drop, then four re-descents, reporting the
     displacement caused by each and the z error each time.
  3. a deliberate xy-offset sweep, which measures the EFFECTIVE CLEARANCE between
     the open pads and the can -- the number that decides whether (b) is
     geometric. The geometric can width is read from MuJoCo as a cross-check,
     because the literal in the code ("the can's 3.3 cm radius") is a comment,
     not a measurement.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_regrasp_after_drop.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent import contracts as C  # noqa: E402
from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

CAN = "alphabet_soup"
OFFSETS_MM = [0, 1, 2, 3, 5, 8, 12]
failures: list[str] = []


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    print(f"[FAIL] {msg}")
    failures.append(msg)


def can_key(objs: dict) -> str:
    return [k for k in objs if CAN in k][0]


def can_xy(sim: LiberoSim) -> np.ndarray:
    objs = sim.pack(sim._obs)["objects"]
    return np.asarray(objs[can_key(objs)], dtype=float)


def width_of(note: str) -> float | None:
    m = re.search(r"width ([0-9.]+)", note)
    return float(m.group(1)) if m else None


def settle(sim: LiberoSim, n: int = 40) -> None:
    for _ in range(n):
        sim.step(np.zeros(C.ACTION_DIM))


def geom_can_extent(sim: LiberoSim) -> str:
    """Read the can's geom footprint straight out of MuJoCo (cross-check)."""
    try:
        mj = sim.env.sim  # robosuite MjSim
        model = mj.model if hasattr(mj, "model") else mj._model
        names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
        idx = [i for i, n in enumerate(names) if CAN in n]
        if not idx:
            return "（读不到 named geom）"
        sizes = [tuple(round(float(v), 4) for v in model.geom_size[i]) for i in idx]
        return f"{len(idx)} geoms: {sizes[:4]}"
    except Exception as exc:  # noqa: BLE001 - a cross-check must not kill the probe
        return f"（读取失败: {type(exc).__name__}: {exc}）"


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    # ---------------------------------------------------------------- geometry
    obs = sim.reset(init_state_idx=0)
    print("can geoms       :", geom_can_extent(sim))
    adapter.set_gripper("open")
    gap_open = adapter._gripper_now()
    print(f"pads fully open : {gap_open:.4f} m  (GRIPPER_OPEN_WIDTH={C.GRIPPER_OPEN_WIDTH})")

    # ------------------------------------------------------------------ 1. base
    obs = sim.reset(init_state_idx=0)
    p0 = can_xy(sim)
    out = adapter.descend_to((p0[0], p0[1]), p0[2] + 0.012)
    p1 = can_xy(sim)
    base_mm = float(np.linalg.norm(p1[:2] - p0[:2])) * 1000.0
    print(f"\n1) baseline descent onto an untouched can: moved {base_mm:.1f} mm  "
          f"(z err {out.note[:32]})")
    # Self-check: the probe must be able to descend WITHOUT disturbing the can,
    # otherwise every number below is measuring its own clumsiness.
    if base_mm < 4.0:
        ok(f"baseline descent is clean ({base_mm:.1f} mm) -> the method is sound")
    else:
        bad(f"baseline descent ALREADY moves the can {base_mm:.1f} mm "
            f"-> all later numbers are suspect")

    # -------------------------------------------------------------- 2. P0e replay
    print("\n2) P0e replay: grasp -> lift 15cm -> open (drop) -> re-descend x4")
    obs = sim.reset(init_state_idx=0)
    p = can_xy(sim)
    adapter.descend_to((p[0], p[1]), p[2] + 0.012)
    g = adapter.set_gripper("close")
    w = width_of(g.note)
    holding = w is not None and C.GRIPPER_EMPTY_WIDTH < w < C.GRIPPER_OPEN_WIDTH
    if holding:
        ok(f"grasped the can (finger width {w:.4f}, between empty "
           f"{C.GRIPPER_EMPTY_WIDTH} and open {C.GRIPPER_OPEN_WIDTH})")
    else:
        bad(f"did NOT grasp the can (note={g.note!r}) -> replay is meaningless")

    z_before = can_xy(sim)[2]
    adapter.move("+z", 15)
    lift_cm = (can_xy(sim)[2] - z_before) * 100.0
    if lift_cm > 10:
        ok(f"payload lifted {lift_cm:.1f} cm with the pads")
    else:
        bad(f"payload only lifted {lift_cm:.1f} cm -> not a real carry")

    adapter.set_gripper("open")
    settle(sim)
    p = can_xy(sim)
    print(f"   after release the can rests at z={p[2]:.4f}")

    for k in range(1, 5):
        before = can_xy(sim)
        out = adapter.descend_to((before[0], before[1]), before[2] + 0.012)
        after = can_xy(sim)
        d_mm = float(np.linalg.norm(after[:2] - before[:2])) * 1000.0
        z_err = re.search(r"z err ([0-9.]+) mm", out.note)
        print(f"   re-descend {k}: can moved {d_mm:6.1f} mm  "
              f"(dy={(after[1] - before[1]) * 1000:+7.1f} mm, "
              f"z_err={z_err.group(1) if z_err else '?':>5} mm, {out.steps} ticks)")

    # --------------------------------------------------------- 3. offset sweep
    print("\n3) effective clearance: descend aimed off-centre by dx")
    print(f"   {'dx':>5} {'can moved':>10}   note")
    for dx_mm in OFFSETS_MM:
        obs = sim.reset(init_state_idx=0)
        p = can_xy(sim)
        out = adapter.descend_to((p[0] + dx_mm / 1000.0, p[1]), p[2] + 0.012)
        q = can_xy(sim)
        d_mm = float(np.linalg.norm(q[:2] - p[:2])) * 1000.0
        print(f"   {dx_mm:>4}mm {d_mm:>9.1f}mm   {out.note[:40]}")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
