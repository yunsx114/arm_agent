"""One-shot summary of an episode against the measured basket opening.

Ground truth used here:
  * basket opening (mesh-accurate, libero_object task 0, init_state 0):
        x in [-0.0665, +0.0957]   y in [+0.1661, +0.3395]
  * a `descend`/`approach` call's requested target_xy IS the object's live
    simulator pose (sim/server.py takes it from pack(obs)["objects"]), so it is a
    ground-truth track of the target object and needs no model narration.
  * a `descend` on the basket aims at the container's BOTTOM centre, so its
    z_offset shows how deep the gripper dove into the mouth.

Usage: python3 scripts/_summarize_episode.py outputs/episodes/episode_XXX.json
"""

from __future__ import annotations

import json
import re
import sys

OPEN_X = (-0.0665, 0.0957)
OPEN_Y = (0.1661, 0.3395)
GRIP_OPEN, GRIP_EMPTY = 0.034, 0.006
DOUBT = ("不在正下方", "没有对准", "未对准", "偏", "还没到", "继续移", "无法落", "不会落进")


def out_of_opening(x: float, y: float) -> tuple[float, float]:
    return (
        max(OPEN_X[0] - x, x - OPEN_X[1], 0.0) * 100,
        max(OPEN_Y[0] - y, y - OPEN_Y[1], 0.0) * 100,
    )


def main() -> int:
    path = sys.argv[1]
    d = json.load(open(path))
    res = d["result"]
    print(f"=== {path}")
    print(f"success={res.get('success')} turns={res.get('turns')} tool_calls={res.get('tool_calls')} "
          f"retries={res.get('protocol_retries')} elapsed={res.get('elapsed_s')}s")

    # --- actor table -------------------------------------------------------
    print("\n=== 逐轮：动作 / 爪末端 / 越出篮口 / 指宽 ===")
    releases: list[tuple[int, float, float]] = []
    closes: list[tuple[int, float, float | None]] = []
    for e in d["transcript"]:
        if "result" not in e:
            continue
        sr = e.get("structured_result") or {}
        tool = sr.get("tool")
        req = sr.get("requested") or {}
        m = sr.get("measurement") or {}
        ee = m.get("eef_end") or m.get("eef_start")
        wm = re.search(r"finger width ([\d.]+)", e["result"])
        w = float(wm.group(1)) if wm else None
        if tool == "gripper" and req.get("action") == "open" and w and w > 0.03:
            x, y = (ee[0], ee[1]) if ee else (float("nan"), float("nan"))
            xo, yo = out_of_opening(x, y)
            tag = "篮口内" if xo == 0 and yo == 0 else "篮口外"
            releases.append((e["turn"], xo, yo))
            print(f"t{e['turn']:>3} OPEN   eef=({x:+.4f},{y:+.4f}) 越出 x{xo:5.1f} y{yo:5.1f}cm  {tag}")
        elif tool == "gripper" and req.get("action") == "close":
            x, y = (ee[0], ee[1]) if ee else (float("nan"), float("nan"))
            kind = "夹住" if w and GRIP_EMPTY < w < GRIP_OPEN else "空夹"
            closes.append((e["turn"], w or 0.0, x))
            print(f"t{e['turn']:>3} CLOSE  eef=({x:+.4f},{y:+.4f}) w={w} {kind}"
                  if w is not None else f"t{e['turn']:>3} CLOSE  no width")

    print(f"\n松手次数={len(releases)}  松手落在篮口内={sum(1 for _, xo, yo in releases if xo == 0 and yo == 0)}")
    print(f"close 次数={len(closes)}  夹住={sum(1 for _, w, _ in closes if GRIP_EMPTY < w < GRIP_OPEN)}"
          f"  空夹={sum(1 for _, w, _ in closes if w <= GRIP_EMPTY)}")

    # --- ground-truth object track from descend targets ---------------------
    print("\n=== 物体真值轨迹（approach 的 target_xy，来自 pack()['objects']）===")
    seen: dict[str, list[tuple[int, float, float, float]]] = {}
    for e in d["transcript"]:
        if "result" not in e:
            continue
        sr = e.get("structured_result") or {}
        if sr.get("tool") not in ("approach", "descend"):
            continue
        req = sr.get("requested") or {}
        xy = req.get("target_xy") or req.get("target")
        if not isinstance(xy, list) or len(xy) < 2:
            continue
        name = str(req.get("target") or req.get("name") or "?")
        z = req.get("z_target")
        seen.setdefault(name, []).append((e["turn"], xy[0], xy[1], z if z is not None else -9))
    for name, rows in seen.items():
        print(f"  {name}:")
        for turn, x, y, z in rows[:6]:
            xo, yo = out_of_opening(x, y)
            print(f"    t{turn:>3} xy=({x:+.4f},{y:+.4f}) z_target={z:+.4f} 越出 x{xo:5.1f} y{yo:5.1f}cm")
        if len(rows) > 6:
            turn, x, y, z = rows[-1]
            xo, yo = out_of_opening(x, y)
            print(f"    …{len(rows)} 次，末次 t{turn} xy=({x:+.4f},{y:+.4f}) 越出 x{xo:5.1f} y{yo:5.1f}cm")

    # --- did the judging step run? ----------------------------------------
    n_sec = {"现状：": 0, "判定：": 0, "动作：": 0}
    doubt = 0
    for e in d["transcript"]:
        if "assistant" not in e:
            continue
        txt = e["assistant"]
        for k in n_sec:
            if k in txt:
                n_sec[k] += 1
        if any(k in txt for k in DOUBT):
            doubt += 1
    print("\n=== 判断环节是否被激活（assistant 文本）===")
    for k, v in n_sec.items():
        print(f"  含 {k}: {v}/80 轮")
    print(f"  含方位存疑措辞: {doubt}/80 轮")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
