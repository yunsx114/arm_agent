"""Where do descend_to's ~1151 ticks go?

MEASURED in M1: one descend from z=0.248 to z=0.050 costs 43 s / 1151 ticks,
but the same 19.8 cm of travel should take ~36 ticks at the servo's measured
~0.55 cm/tick (probe_action_semantics: 5 cm in 9 ticks). 30x the necessary
ticks means something is looping. This probe monkeypatches ActionAdapter.move
and logs every inner call (direction, requested cm, ticks, resulting delta) so
the waste is measured — repeated xy corrections? stuck-detector stalls?
overshoot? — instead of guessed.

    cd arm_agent && bash scripts/simenv.sh scripts/probe_descend_cost.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    obs = sim.reset(init_state_idx=0)
    adapter = ActionAdapter(sim)

    objs = obs["objects"]
    key = [k for k in objs if "soup" in k][0]
    pos = objs[key]
    print(f"target {key} pos={pos}")

    calls: list[tuple] = []
    orig = adapter.move

    def wrapped(direction: str, distance_cm: float):
        out = orig(direction, distance_cm)
        calls.append(
            (direction, round(distance_cm, 2), out.steps, [round(v, 4) for v in out.measured_delta_m])
        )
        return out

    adapter.move = wrapped  # type: ignore[assignment]

    eef0 = adapter._eef_now().copy()
    z_target = pos[2] + 0.012
    out = adapter.descend_to((pos[0], pos[1]), z_target)
    eef1 = adapter._eef_now()

    print(f"\ndescend_to total: {out.steps} ticks, note={out.note!r}")
    print(f"eef {np.round(eef0, 4).tolist()} -> {np.round(eef1, 4).tolist()} (z_target {z_target:.4f})")
    print(f"\ninner move() calls: {len(calls)}")

    by_dir = Counter(c[0] for c in calls)
    ticks_by_dir: Counter = Counter()
    for d, _cm, t, _delta in calls:
        ticks_by_dir[d] += t
    print(f"{'dir':>4} {'calls':>6} {'ticks':>7} {'avg ticks/call':>15}")
    for d in ("+x", "-x", "+y", "-y", "+z", "-z"):
        if by_dir[d]:
            print(f"{d:>4} {by_dir[d]:>6} {ticks_by_dir[d]:>7} {ticks_by_dir[d] / by_dir[d]:>15.1f}")
    print(f"  total ticks inside move(): {sum(c[2] for c in calls)}")
    print(f"  zero-tick calls: {sum(1 for c in calls if c[2] == 0)}")

    print("\nfirst 20 calls (dir, cm, ticks, delta):")
    for c in calls[:20]:
        print("  ", c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
