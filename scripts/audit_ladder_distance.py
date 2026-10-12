#!/usr/bin/env python3
"""How close did the gripper actually get to the can? (raw baseline vs 档 D ladder)

THE ONE NUMBER THAT IS COMPARABLE. `raw` mode shows the model nothing about object
positions, so "which object did it aim at" cannot be audited from tool output. But every
motion reports the END EFFECTOR's own position, and the objects' truth for each init state
is fixed (`scripts/probe_init_positions.py`), so the minimum EEF-to-can distance over an
episode is measurable for BOTH arms in exactly the same way.

The grasp tolerance is 2.2cm: below it, closing the pads can hold the can; above it, the
pads close on air. So this number is the difference between "was in range" and "was not",
which is the only thing a 0/8 vs 0/8 comparison can honestly report.

USAGE
    python3 scripts/audit_ladder_distance.py outputs/episodes/episode_00*.json
    python3 scripts/audit_ladder_distance.py --truth outputs/init_truth.json <files...>
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TRUTH = REPO / "outputs" / "init_truth.json"
GRASP_TOL_M = 0.022  # MEASURED: 2.2cm grasps, 4.6cm closes on air


def eef_track(path: Path) -> list[list[float]]:
    """Every end-effector position the episode's own tool results report."""
    entries = (json.loads(path.read_text()).get("transcript") or [])
    track = []
    for entry in entries:
        measurement = (entry.get("structured_result") or {}).get("measurement") or {}
        for key in ("eef_end", "eef_start"):
            pos = measurement.get(key)
            if isinstance(pos, list) and len(pos) == 3:
                track.append([float(v) for v in pos])
                break
    return track


def main(argv: list[str]) -> int:
    truth_path = DEFAULT_TRUTH
    args = []
    it = iter(argv[1:])
    for a in it:
        if a == "--truth":
            truth_path = Path(next(it))
        else:
            args.append(a)
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}
    paths = [Path(p) for a in args for p in sorted(glob.glob(a))]
    if not paths:
        print("no episodes given")
        return 1

    print(f"{'episode':<30}{'init':>5}{'移动':>6}{'XY最近':>9}{'<22mm?':>8}"
          f"{'3D最近':>9}{'结束时XY':>10}{'夹住':>5}{'成功':>5}")
    best_all = []
    best_xy_all = []
    for path in paths:
        parts = path.stem.split("_")  # episode_000_1791777689 -> ['episode','000','<ts>']
        # The run used INIT_STATE=0 and `init_idx = (INIT_STATE + episode_id) % 50`
        # (cli.cmd_run), so the episode id IS the init state.
        init_idx = int(parts[1])
        can = (truth.get(str(init_idx)) or {}).get("can")
        track = eef_track(path)
        if not can or not track:
            print(f"{path.name:<30} (缺真值或轨迹)")
            continue
        dists = [((p[0] - can[0]) ** 2 + (p[1] - can[1]) ** 2 + (p[2] - can[2]) ** 2) ** 0.5
                 for p in track]
        # XY separately: the grasp is an XY alignment plus a descent, and the EEF origin
        # sits ~2.4cm ABOVE the pads' grasp point (MEASURED: closing on the can reports
        # eef z=0.061 while the can's body centre is z=0.038). Mixing that constant offset
        # into a 3-D distance hides how well the model actually AIMED, so both are shown.
        xy = [((p[0] - can[0]) ** 2 + (p[1] - can[1]) ** 2) ** 0.5 for p in track]
        best = min(dists)
        best_xy = min(xy)
        best_all.append(best)
        best_xy_all.append(best_xy)
        data = json.loads(path.read_text())
        result = data.get("result") or {}
        grasps = sum(
            1 for e in (data.get("transcript") or [])
            if (((e.get("structured_result") or {}).get("status")) == "grasped")
        )
        print(f"{path.name:<30}{init_idx:>5}{len(track):>6}{best_xy * 1000:>8.0f}mm"
              f"{'✓' if best_xy <= GRASP_TOL_M else '·':>8}"
              f"{best * 1000:>8.0f}mm{xy[-1] * 1000:>9.0f}mm{grasps:>5}"
              f"{'✓' if result.get('success') else '·':>5}")
    if best_all:
        print()
        n = len(best_all)
        print(f"{n} 回合：XY 最近 中位 {sorted(best_xy_all)[n // 2] * 1000:.0f}mm、"
              f"最小 {min(best_xy_all) * 1000:.0f}mm、进入 22mm 容差的 {sum(1 for b in best_xy_all if b <= GRASP_TOL_M)}/{n}；"
              f"3D 最近 中位 {sorted(best_all)[n // 2] * 1000:.0f}mm、最小 {min(best_all) * 1000:.0f}mm")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
