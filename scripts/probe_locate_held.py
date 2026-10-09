"""Unit-check the `locate` reply for the case that broke v8.

v8 (turns 64-80) looped `move -z` -> `descend_to(soup)` -> `close` -> `move +z 15`
for 17 turns without ever approaching the basket, because `locate alphabet_soup`
answered with the GRIPPER's own coordinates (the can was in the pads) and the
model read that as "the can is resting there".

`_locate` does not use `self`, so it can be exercised with a hand-built obs --
no simulator, no GPU. Self-check: the same reply WITHOUT the object in the
gripper must NOT contain the new warning (otherwise the warning is unconditional
noise and the model would learn to ignore it).

Run: PYTHONPATH=src python scripts/probe_locate_held.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.agent.harness import Harness  # noqa: E402

SOUP = [0.0312, 0.2718, 0.0280]
BASKET = [0.0148, 0.2521, -0.0046]

CASES = [
    ("holding it (eef inside the can)",
     {"objects": {"alphabet_soup_1": SOUP, "basket_1": BASKET},
      "eef_pos": [0.0253, 0.2744, 0.0300]}, True),
    ("not holding (eef high above)",
     {"objects": {"alphabet_soup_1": SOUP, "basket_1": BASKET},
      "eef_pos": [0.0, 0.0, 0.25]}, False),
]


def main() -> int:
    h = object.__new__(Harness)  # bypass __init__: _locate needs only self._grasped
    failures = 0
    for label, obs, expect_warn in CASES:
        # "in the pads" is now a fact about the GRIPPER (set by a successful
        # `set_gripper close`), not about distance -- see the P0a false-alarm fix.
        h._grasped = expect_warn
        text = h._locate("alphabet_soup", obs)
        has = "它目前在夹爪里" in text
        ok = has == expect_warn
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: held-warning={has} (expected {expect_warn})")

        # P0b contract: a tool result may carry MEASUREMENTS, never PRESCRIPTIONS.
        # Both halves were observed live:
        #   - v9: "下一步建议: 先 move +x 1.6cm" (computed from the object riding
        #     the pads) ordered a move while the can was already in the hand;
        #   - P0a' t11: "x 轴已对齐；y 轴已对齐" announced alignment was finished
        #     while dz was still -5.5cm, so the model never descended.
        prescribed = [w for w in ("下一步", "建议", "不要", "先 move", "抬到", "篮口") if w in text]
        ok2 = not prescribed
        failures += 0 if ok2 else 1
        print(f"[{'PASS' if ok2 else 'FAIL'}]   no prescriptions in the reply: found={prescribed}")
        if has:
            for line in text.splitlines():
                print(f"        {line.strip()}")
    print()
    print()
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
