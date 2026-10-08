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
    failures = 0
    for label, obs, expect_warn in CASES:
        text = Harness._locate(None, "alphabet_soup", obs)
        has = "它正被你的夹爪抓着" in text
        ok = has == expect_warn
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: held-warning={has} (expected {expect_warn})")

        # The v9 failure mode: the reply offered BOTH "下一步建议: 先 move +x …"
        # (derived from the object riding the pads -- pure noise) AND a warning
        # to ignore those numbers. The model obeyed the first line and drove
        # away from the basket. The two must never coexist.
        contradicts = expect_warn and ("下一步建议" in text or "抓取高度" in text)
        ok2 = not contradicts
        failures += 0 if ok2 else 1
        print(f"[{'PASS' if ok2 else 'FAIL'}]   self-consistent (no 下一步建议/抓取高度 when held): "
              f"contradicts={contradicts}")
        if has:
            for line in text.splitlines():
                print(f"        {line.strip()}")
    print()
    print("=== height gate (carry high enough to clear the rim?) ===")
    for label, z, expect_ok in [("low: centre z=0.028", 0.028, False),
                                ("high: centre z=0.20", 0.20, True)]:
        obs = {
            "objects": {"alphabet_soup_1": [0.0312, 0.2718, z], "basket_1": BASKET},
            "eef_pos": [0.0253, 0.2744, z + 0.02],
        }
        text = Harness._locate(None, "alphabet_soup", obs)
        is_ok = "高度 OK" in text
        ok = is_ok == expect_ok
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: height_ok={is_ok} (expected {expect_ok})")

    print()
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
