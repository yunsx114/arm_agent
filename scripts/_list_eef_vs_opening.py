"""EEF x/y at every turn, against the basket opening box.

Measured basket opening (mesh-accurate, libero_object task 0, init_state 0):
    x in [-0.0665, +0.0957]   y in [+0.1661, +0.3395]   centre (0.0146, 0.2528)
"""

from __future__ import annotations

import json
import sys

EPISODE = sys.argv[1] if len(sys.argv) > 1 else "outputs/episodes/episode_000_1791623248.json"
OPEN_X = (-0.0665, 0.0957)
OPEN_Y = (0.1661, 0.3395)

d = json.load(open(EPISODE))
print(f"{'turn':>4} {'tool':<14} {'eef_x':>9} {'eef_y':>9}  x_out  y_out  inside")
for e in d["transcript"]:
    if "result" not in e:
        continue
    sr = e.get("structured_result") or {}
    m = sr.get("measurement") or {}
    ee = m.get("eef_end")
    if not ee:
        continue
    x, y = ee[0], ee[1]
    xo = max(OPEN_X[0] - x, x - OPEN_X[1], 0.0) * 100
    yo = max(OPEN_Y[0] - y, y - OPEN_Y[1], 0.0) * 100
    inside = "YES" if (xo == 0 and yo == 0) else "no"
    print(
        f"{e['turn']:>4} {str(sr.get('tool')):<14} {x:>+9.4f} {y:>+9.4f} "
        f"{xo:>6.1f} {yo:>6.1f}  {inside}"
    )
