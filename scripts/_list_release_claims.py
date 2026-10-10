"""List the turns where the model asserted it was above the basket.

Single-purpose helper for the release-point audit: it prints the assistant's own
words next to the measured EEF y, so the claim/turn pairs can be turned into a
frame-extraction list for probe_release_perception.py.
"""

from __future__ import annotations

import json
import re
import sys

EPISODE = sys.argv[1] if len(sys.argv) > 1 else "outputs/episodes/episode_000_1791623248.json"
KEYWORDS = ("正上方", "上方", "篮口", "篮子口", "对准", "放下", "落入", "松开", "释放", "框中", "筐")

d = json.load(open(EPISODE))
t = d["transcript"]

eef_y: dict[int, float] = {}
for e in t:
    if "result" in e:
        m = (e.get("structured_result") or {}).get("measurement") or {}
        ee = m.get("eef_end")
        if ee:
            eef_y[e["turn"]] = ee[1]

print(f"{'turn':>4} {'eef_y':>9}  claim")
for e in t:
    if "assistant" not in e:
        continue
    txt = e["assistant"]
    body = txt.split("<tool_call>")[0].strip()
    if not body:
        continue
    if any(k in body for k in KEYWORDS):
        y = eef_y.get(e["turn"] - 1)
        ys = f"{y:+.4f}" if y is not None else "   -   "
        print(f"{e['turn']:>4} {ys:>9}  {body[:110]!r}")

# Which turns were followed by an actual open (release)?
print("\n=== release turns (open with width>0.03) vs basket y=+0.2528, opening x[-0.0665,+0.0957] y[+0.1661,+0.3395] ===")
for e in t:
    if "result" not in e:
        continue
    sr = e.get("structured_result") or {}
    if sr.get("tool") != "gripper" or (sr.get("requested") or {}).get("action") != "open":
        continue
    m = sr.get("measurement") or {}
    ee = m.get("eef_end")
    w = re.search(r"finger width ([\d.]+)", e["result"])
    if not ee or not w or float(w.group(1)) < 0.03:
        continue
    x, y = ee[0], ee[1]
    inside = (-0.0665 <= x <= 0.0957) and (0.1661 <= y <= 0.3395)
    print(
        f"  t{e['turn']:>3} eef=({x:+.4f},{y:+.4f})  x_out={max(-0.0665 - x, x - 0.0957, 0) * 100:5.1f}cm"
        f" y_out={max(0.1661 - y, y - 0.3395, 0) * 100:5.1f}cm  {'篮口内' if inside else '篮口外'}"
    )
