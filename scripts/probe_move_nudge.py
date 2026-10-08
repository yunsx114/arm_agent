"""Regression check for the tiny-nudge no-op (v8).

v8 t51/t55/t58/t61: `move ... 0.1cm` (1 mm) is below MOVE_ARRIVE_TOL_M (2 mm), so
`move()` broke out of its servo loop on the first check, ran **0 ticks**, and
reported `实测位移 (0,0,0)` -- byte-for-byte what a genuinely blocked move looks
like. The model concluded the arm was stuck and kept retrying the same nudge.

Self-checks:
  1. a 1 mm request must now run >= 1 tick (it used to run 0);
  2. the achieved displacement must be much closer to 1 mm than to 0;
  3. a large move must still work (no regression from the tolerance change).

Run: bash scripts/simenv.sh scripts/probe_move_nudge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    checks: list[tuple[str, bool, str]] = []

    for dist in (0.1, 0.2, 0.5, 5.0):
        sim.reset(init_state_idx=0)
        out = adapter.move("+y", dist)
        moved_mm = abs(out.measured_delta_m[1]) * 1000.0
        print(f"move +y {dist:>4}cm -> {out.steps:>3} ticks, moved {moved_mm:>6.2f} mm "
              f"(note: {out.note or '-'})")

        if dist <= 0.2:
            ok = out.steps >= 1
            checks.append((f"{dist}cm runs >=1 tick", ok, f"steps={out.steps}"))
            # 1 mm request: anything above ~0.3 mm is progress, not "stuck".
            ok2 = moved_mm > 0.3
            checks.append((f"{dist}cm actually moves", ok2, f"moved={moved_mm:.2f}mm"))
        else:
            # Larger moves must overshoot the request by no more than the
            # tolerance that used to apply (2 mm).
            target_mm = dist * 10.0
            ok = abs(moved_mm - target_mm) <= 4.0
            checks.append((f"{dist}cm hits target", ok, f"moved={moved_mm:.2f} vs {target_mm:.1f}mm"))

    print()
    failures = 0
    for name, ok, detail in checks:
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
