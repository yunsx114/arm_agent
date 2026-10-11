"""Audit every release in an episode: was the grip actually carrying, and where
was the basket when the fingers opened?

WHY this exists: the harness now logs its own arithmetic on every carrying
release (`[turn N] 放手位置自检: ...`), but that line lives in a log file that the
next run overwrites -- while the transcripts are permanent. This script
re-derives the same numbers from the transcripts, so two arms (or a run made
before the instrument existed) can be audited the same way:

  * the release turn is a `[set_gripper open]` whose finger width is the OPEN
    width AND whose sequence was preceded by a `close` that reported `holding`
    (an empty-hand open reads the same text and must not be counted);
  * the EEF position comes from the tool result text of that same turn;
  * the basket's centre comes from the most recent `approach(basket)`'s
    `requested.target_xy` (the sim's own measurement at that moment).

Usage:  python3 scripts/audit_release_poses.py outputs/episodes/episode_*.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OPEN_W = 0.034        # >= this finger width is an open gripper
RIM_Z = 0.1513        # measured rim height (probe_rim.py)
CAN_BOTTOM = 0.050    # EEF -> payload bottom while carrying
OPENING_HALF = {"x": 0.0811, "y": 0.0867}   # half-extents of the opening rect


def audit(path: str) -> None:
    d = json.load(open(path, encoding="utf-8"))
    name = Path(path).stem
    turns = d.get("transcript", [])
    success = (d.get("result") or {}).get("success")
    print(f"\n=== {name}  success={success} ===")

    basket = None
    pending = None
    carried = False          # rebuilt from the tool results, like the harness's _grasped
    n_rel = n_ok = 0
    for e in turns:
        # The transcript stores TWO entries per turn (assistant, then result) but a
        # result entry carries `structured_result` as well -- an elif chain silently
        # drops the `result` branch there (this cost one debugging round; the same
        # trap bit the ad-hoc analysis of episode_000_1791722140).
        if "assistant" in e:
            a = e["assistant"]
            m = re.search(r"<function=approach>.*?<parameter=target>\s*([^<]*?)\s*</parameter>", a, re.S)
            pending = m.group(1).strip() if m else None
            continue

        sr = e.get("structured_result") or {}
        req = sr.get("requested") or {}
        if sr.get("tool") == "approach" and req.get("target_xy") and pending and "basket" in pending:
            basket = (float(req["target_xy"][0]), float(req["target_xy"][1]))
        pending = None

        if "result" not in e:
            continue
        r = e["result"]
        m = re.search(r"finger width ([\d.]+) \((\w+)\)", r)
        if not m:
            continue
        width, kind = float(m.group(1)), m.group(2)
        # The result text reads "(指间无物体)" both for a release and for an
        # empty-hand open, so the only honest signal is the SEQUENCE: a close that
        # reported `holding` is what makes the next open a release.
        if kind == "holding":
            carried = True
            continue
        if kind != "open" or not carried:
            continue
        carried = False
        n_rel += 1
        ee = re.search(r"机械爪\s*位于\s*\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)", r)
        if ee and basket:
            x, y, z = (float(ee.group(i)) for i in (1, 2, 3))
            dx, dy = (x - basket[0]) * 100, (y - basket[1]) * 100
            bot = z - CAN_BOTTOM
            inside = abs(dx) <= 5 and abs(dy) <= 5
            clear = bot >= RIM_Z
            ok = inside and clear
            n_ok += ok
            why = "" if ok else "  <<< " + ("不在口内 " if not inside else "") + ("低于口沿" if not clear else "")
            print(f"  t{e.get('turn'):>2} 放手 爪=({x:+.4f},{y:+.4f},{z:.4f}) "
                  f"篮口=({basket[0]:+.4f},{basket[1]:+.4f}) Δ=({dx:+.1f},{dy:+.1f})cm "
                  f"物体底={bot:.4f}{why}")
        elif ee and not basket:
            print(f"  t{e.get('turn'):>2} 放手 爪=({ee.group(1)},{ee.group(2)},{ee.group(3)}) 但之前没有 approach(basket)，无法算 Δ")
    print(f"  合计：夹着东西放手 {n_rel} 次，其中位置正确 {n_ok} 次"
          + ("" if n_rel else "（本 episode 没有夹着东西放手）"))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        audit(p)
