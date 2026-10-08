"""Unit-check the two new outcome guards added after v10.

Both guard a failure the model literally could not see:

  (a) `move` reported a short displacement with **no** indication at all -- the
      adapter's "blocked" note was being dropped in format_outcome, so a stalled
      descent looked exactly like a successful one (v10 t47: asked -7.5 cm,
      moved 0.75 cm, silently).
  (b) `descend_to` announced success -- "已对准并下降到 … 下一步用 set_gripper
      操作夹爪" -- while parked 6.6 cm above the can (v10 t28/t36: `z err 65.9
      mm`), which is why the model closed on air 71 turns in a row.

Self-check: the same shapes WITHOUT the shortfall must NOT carry the warning.
Otherwise the warning is unconditional noise and the model learns to ignore it.

Run: PYTHONPATH=src python scripts/probe_outcome_guards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.agent.harness import _z_err, format_outcome  # noqa: E402

MOVE_CASES = [
    (
        "v10 t47: asked -z 7.5cm, got 0.75cm, adapter said blocked",
        {
            "kind": "move",
            "requested": {"direction": "-z", "distance_cm": 7.5},
            "measured_delta_m": [0.0079, -0.0008, -0.0075],
            "steps": 8,
            "note": "blocked on -z: commanded -5.0cm/tick but moved <0.5mm for 5 ticks",
            "eef_end": [0.0132, 0.1351, 0.1002],
        },
        True,
    ),
    (
        "healthy move: asked +y 15cm, got 15.1cm",
        {
            "kind": "move",
            "requested": {"direction": "+y", "distance_cm": 15},
            "measured_delta_m": [0.0041, 0.1509, 0.0075],
            "steps": 15,
            "note": "gentle carry (1.0cm/tick, 1.00 along the opening axis)",
            "eef_end": [-0.0001, 0.1046, 0.4931],
        },
        False,
    ),
    (
        "healthy tiny nudge: asked 0.1cm (under tolerance, must NOT cry blocked)",
        {
            "kind": "move",
            "requested": {"direction": "+y", "distance_cm": 0.1},
            "measured_delta_m": [0.0, 0.0007, 0.0],
            "steps": 3,
            "note": "gentle carry (1.0cm/tick, 1.00 along the opening axis)",
            "eef_end": [0.0, 0.1, 0.25],
        },
        False,
    ),
]

Z_ERR_CASES = [
    ("v10 t28 note", "xy err 1.8 mm, z err 65.9 mm", 0.0659),
    ("v9 good note", "xy err 2.8 mm, z err 4.8 mm", 0.0048),
    ("no z err", "blocked on -z", None),
]


def main() -> int:
    failures = 0

    print("=== move shortfall guard ===")
    for label, outcome, expect in MOVE_CASES:
        text = format_outcome(outcome)
        has = "只走了" in text
        ok = has == expect
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        print(f"        warning={has} (expected {expect})")
        if has:
            print(f"        {text}")

    print()
    print("=== _z_err parsing ===")
    for label, note, expect in Z_ERR_CASES:
        got = _z_err(note)
        ok = (got is None and expect is None) or (
            got is not None and expect is not None and abs(got - expect) < 1e-9
        )
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {got} (expected {expect})")

    print()
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
