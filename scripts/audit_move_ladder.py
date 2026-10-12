#!/usr/bin/env python3
"""Audit a 档 D run: did the model USE the distance ladder, or fight it?

The ladder exists to remove ONE thing from the model's job: turning "10-12 px to the
left" into a world distance in centimetres. It leaves the model two decisions -- a
direction and a granularity -- plus the rule the prompt states: overshoot -> downshift,
too little -> upshift.

So the questions this script answers, all from the transcript alone:

1. Which rungs were used? (`far`/`near`/`fine` counts)
2. Does it DOWN-SHIFT after an overshoot? An overshoot is measurable without truth: the
   model reverses axis direction, which means the previous step went past the target.
   The prompt's rule says the rung after a reversal should be smaller. Reported as
   downshift / same / upshift, per rung.
3. Does it CONTRADICT ITSELF? The model writes its 判定 before it calls, so its text says
   what it thinks the step is ("一大步", "一小步", "稍微"). When the text says small and
   the call is `far`, the ladder did not fail -- the model's judgement of magnitude did,
   and that is a different bug with a different fix. This is the highest-value number in
   the file.
4. Is it oscillating? Longest alternating run of reversals, and the share of repeated
   identical calls.

USAGE
    python3 scripts/audit_move_ladder.py outputs/episodes/episode_0*.json
"""
from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

RUNG_ORDER = {"fine": 0, "near": 1, "far": 2}
RUNG_CM = {"fine": 0.1, "near": 1.0, "far": 10.0}

# Intent phrases, split by the magnitude they claim. Deliberately literal: the model
# writes "向 -y 走一大步（10cm）" or "回退一小步（1cm）" in the same sentence as the call,
# so this is a copy of its own words, not an interpretation.
BIG_WORDS = re.compile(r"一大步|走一大步|大步|较大幅度|10\s*cm|10\s*厘米|十几厘米|很远")
SMALL_WORDS = re.compile(r"一小步|小步|微调|稍微|一点点?|很小|毫米|mm|1\s*cm|1\s*厘米|细微")

# The model quotes its own pixel error in the same sentence ("在爪心的 -y 方向约 10~12
# 像素"). The prompt hands it the scale (1px ≈ 3-4mm in the global view, MEASURED), so
# that quote IS a distance estimate -- which makes it possible to ask the question the
# ladder is really about: is the rung it picked bigger than the error it is fixing?
# A step twice its own error cannot converge, however good its direction is.
PX_CLAIM = re.compile(r"(\d+)\s*[~～\-到至]\s*(\d+)\s*像素|约\s*(\d+)\s*像素|(\d+)\s*像素")
GLOBAL_MM_PER_PX = 3.7  # MEASURED: 2.96-4.04 mm/px across the workspace (agentview)

# GROUND TRUTH for the can, one entry per init_state (`probe_init_positions.py`).
# The can sits at the same spot for init states 0-7; only the basket moves, so the
# table is small and stable. Used to answer the question the whole 档 D line is about:
# the model's own "I am still N cm away" versus how far away it really is.
TRUTH_PATH = Path(__file__).resolve().parents[1] / "outputs" / "init_truth.json"


def can_xy(init_idx: int | None) -> tuple[float, float] | None:
    if init_idx is None or not TRUTH_PATH.exists():
        return None
    try:
        entry = json.loads(TRUTH_PATH.read_text()).get(str(init_idx))
    except (OSError, ValueError):
        return None
    can = (entry or {}).get("can")
    return (float(can[0]), float(can[1])) if can else None


def calls_in(path: Path) -> list[dict]:
    """Every tool call in an episode, with the text the model wrote before it."""
    data = json.loads(path.read_text())
    entries = data.get("transcript") or []
    calls: list[dict] = []
    pending_text = ""
    for entry in entries:
        assistant = entry.get("assistant")
        if isinstance(assistant, str):
            pending_text = re.sub(r"<.*?>", " ", assistant.split("<tool_call>")[0]).strip()
            continue
        if not entry.get("result"):
            continue
        sr = entry.get("structured_result") or {}
        tool = sr.get("tool") or entry.get("tool") or "?"
        requested = sr.get("requested") or {}
        measurement = sr.get("measurement") or {}
        calls.append({
            "turn": entry.get("turn"),
            "tool": tool,
            "text": pending_text,
            "direction": str(requested.get("direction") or ""),
            "rung": measurement.get("move_size"),
            "distance_cm": requested.get("distance_cm"),
            "eef_end": measurement.get("eef_end"),
            "status": sr.get("status"),
            "result": str(entry.get("result") or ""),
        })
        pending_text = ""
    return calls


def declared_magnitude(text: str) -> str:
    """What the model's own 判定 claims the step size is: big / small / unknown."""
    big = bool(BIG_WORDS.search(text))
    small = bool(SMALL_WORDS.search(text))
    if big and small:  # e.g. "先走一大步，再一小步微调" -- the call decides which it meant
        return "both"
    if big:
        return "big"
    if small:
        return "small"
    return "unknown"


def claimed_error_cm(text: str) -> float | None:
    """The distance the model itself claims remains, from the px error it quotes.

    Uses the MIDDLE of a quoted range ("10~12 像素" -> 11px) and the measured global
    scale. Returns None when the model did not quote a pixel error -- which is itself
    worth knowing, because the prompt tells it to read the image every step.
    """
    m = PX_CLAIM.search(text)
    if not m:
        return None
    lo, hi = m.group(1), m.group(2)
    px = (float(lo) + float(hi)) / 2.0 if lo and hi else float(m.group(3) or m.group(4))
    return px * GLOBAL_MM_PER_PX / 10.0


def effective_rung(call: dict) -> str | None:
    """The magnitude bucket of a move.

    With the ladder on this is literally what the model asked for. Without it (`raw`
    baseline episodes) the model picked a free distance, so it is binned on the same
    10x scale the ladder uses: <= 0.5cm behaves like `fine`, <= 5cm like `near`, else
    `far`. That is an approximation and is only for putting the two arms on one axis --
    the baseline was free to write 0.4 or 4.9, and this rounds both to `near`.
    """
    if call.get("rung") in RUNG_ORDER:
        return call["rung"]
    try:
        cm = abs(float(call.get("distance_cm")))
    except (TypeError, ValueError):
        return None
    if cm <= 0.5:
        return "fine"
    if cm <= 5.0:
        return "near"
    return "far"


def summarise(path: Path, verbose: bool = False) -> dict:
    calls = calls_in(path)
    moves = []
    for c in calls:
        if c["tool"] != "move":
            continue
        rung = effective_rung(c)
        if rung:
            moves.append({**c, "rung": rung})
    counts = collections.Counter(c["rung"] for c in moves)

    # Reversals and the rung chosen right after one.
    after_reversal = collections.Counter()
    reversals = 0
    for prev, cur in zip(moves, moves[1:]):
        if not prev["direction"] or not cur["direction"]:
            continue
        if prev["direction"] == cur["direction"]:
            continue
        if prev["direction"][1:] != cur["direction"][1:]:
            continue  # a different axis, not a reversal
        reversals += 1
        if RUNG_ORDER[cur["rung"]] < RUNG_ORDER[prev["rung"]]:
            after_reversal["downshift"] += 1
        elif RUNG_ORDER[cur["rung"]] == RUNG_ORDER[prev["rung"]]:
            after_reversal["same"] += 1
        else:
            after_reversal["upshift"] += 1

    # Text vs call.
    mismatch = {"text_small_call_far": 0, "text_big_call_fine": 0}
    overshoot = []  # rung_cm / claimed_error_cm, per move that quoted a px error
    quoted = 0
    for c in moves:
        said = declared_magnitude(c["text"])
        if said == "small" and c["rung"] == "far":
            mismatch["text_small_call_far"] += 1
        if said == "big" and c["rung"] == "fine":
            mismatch["text_big_call_fine"] += 1
        error_cm = claimed_error_cm(c["text"])
        if error_cm and error_cm > 0.05:
            quoted += 1
            overshoot.append(RUNG_CM[c["rung"]] / error_cm)
            c["error_cm"] = round(error_cm, 2)
            c["overshoot"] = round(RUNG_CM[c["rung"]] / error_cm, 2)
    overshoot.sort()
    median = overshoot[len(overshoot) // 2] if overshoot else 0.0
    over2 = sum(1 for v in overshoot if v >= 2.0)

    # CLAIMED vs TRUE DISTANCE. The px quote is the model's own estimate of how far it
    # still has to go; the truth comes from the can's recorded position and the end
    # effector's own reported position. If the model's estimate is systematically SMALL,
    # it will pick a rung that is too small EVERY time -- and that is not a ladder bug,
    # it is an estimator bug the ladder merely exposes.
    init_idx = None
    parts = path.stem.split("_")
    if len(parts) >= 2 and parts[0] == "episode":
        try:
            init_idx = int(parts[1])
        except ValueError:
            init_idx = None
    truth_xy = can_xy(init_idx)
    ratios: list[float] = []
    if truth_xy:
        for c in moves:
            eef = c.get("eef_end")
            claim_px = PX_CLAIM.search(c["text"])
            if not (isinstance(eef, list) and len(eef) == 3 and claim_px):
                continue
            lo, hi = claim_px.group(1), claim_px.group(2)
            px = (float(lo) + float(hi)) / 2.0 if lo and hi else float(claim_px.group(3) or claim_px.group(4))
            claimed_cm = px * GLOBAL_MM_PER_PX / 10.0
            true_cm = (((float(eef[0]) - truth_xy[0]) ** 2 + (float(eef[1]) - truth_xy[1]) ** 2) ** 0.5) * 100.0
            if true_cm > 0.5:
                ratios.append(claimed_cm / true_cm)
    ratio_median = sorted(ratios)[len(ratios) // 2] if ratios else 0.0
    under2 = sum(1 for v in ratios if v < 0.5)

    # Longest alternating run (ping-pong).
    longest = run = 0
    last_dir = None
    for c in moves:
        d = c["direction"]
        if last_dir and d and d[1:] == last_dir[1:] and d[0] != last_dir[0]:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
        last_dir = d

    grasps = [c for c in calls if c["tool"] == "gripper" and c["status"]]
    verdicts = collections.Counter(c["status"] for c in grasps)
    identical = sum(
        1 for prev, cur in zip(moves, moves[1:]) if (prev["direction"], prev["rung"]) == (cur["direction"], cur["rung"])
    )
    summary = {
        "file": path.name,
        "moves": len(moves),
        "rungs": dict(counts),
        "reversals": reversals,
        "after_reversal": dict(after_reversal),
        "mismatch": mismatch,
        "quoted": quoted,
        "overshoot_median": median,
        "overshoot_over2": over2,
        "dist_ratios": len(ratios),
        "dist_ratio_median": ratio_median,
        "dist_under2": under2,
        "longest_pingpong": longest,
        "repeat_calls": identical,
        "gripper": dict(verdicts),
        "success": bool((json.loads(path.read_text()).get("result") or {}).get("success")),
    }
    if verbose:
        for c in moves:
            claim = f"自称误差 {c['error_cm']}cm 过冲 {c['overshoot']}x" if c.get("overshoot") else ""
            print(f"   t{c['turn']:>3} {c['direction']:>3} {str(c['rung']):>5} "
                  f"({RUNG_CM.get(c['rung'], 0)}cm) 说的话: {declared_magnitude(c['text'])} {claim}")
    return summary


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    verbose = "--verbose" in argv
    paths = [Path(p) for a in args for p in sorted(glob.glob(a))] or \
            [Path(p) for p in sorted(glob.glob("outputs/episodes/episode_0*.json"))]
    if not paths:
        print("no episodes found")
        return 1

    totals = collections.Counter()
    after_total = collections.Counter()
    mismatch_total = collections.Counter()
    quoted_total = over2_total = 0
    overshoots: list[float] = []
    ratio_total = under2_total = 0
    ratio_medians: list[float] = []
    print(f"{'episode':<28}{'moves':>6}{'far':>5}{'near':>6}{'fine':>6}"
          f"{'反转':>5}{'降档':>5}{'同档':>5}{'升档':>5}"
          f"{'自称像素':>9}{'过冲中位':>9}{'过冲>=2x':>10}"
          f"{'自称/真值':>10}{'低估>=2x':>9}{'最长乒乓':>9}{'夹住':>5}{'成功':>5}")
    for path in paths:
        s = summarise(path, verbose=verbose)
        totals.update(s["rungs"])
        after_total.update(s["after_reversal"])
        mismatch_total.update(s["mismatch"])
        quoted_total += s["quoted"]
        over2_total += s["overshoot_over2"]
        overshoots.append(s["overshoot_median"])
        ratio_total += s["dist_ratios"]
        under2_total += s["dist_under2"]
        if s["dist_ratios"]:
            ratio_medians.append(s["dist_ratio_median"])
        print(f"{s['file']:<28}{s['moves']:>6}{s['rungs'].get('far', 0):>5}"
              f"{s['rungs'].get('near', 0):>6}{s['rungs'].get('fine', 0):>6}"
              f"{s['reversals']:>5}{s['after_reversal'].get('downshift', 0):>5}"
              f"{s['after_reversal'].get('same', 0):>5}{s['after_reversal'].get('upshift', 0):>5}"
              f"{s['quoted']:>9}{s['overshoot_median']:>9.2f}"
              f"{s['overshoot_over2']:>10}{s['dist_ratio_median']:>10.2f}"
              f"{s['dist_under2']:>9}{s['longest_pingpong']:>9}"
              f"{s['gripper'].get('grasped', 0):>5}{'✓' if s['success'] else '·':>5}")
    print()
    print(f"合计：{len(paths)} 回合，移动 {sum(totals.values())} 次  "
          f"far={totals.get('far', 0)} near={totals.get('near', 0)} fine={totals.get('fine', 0)}")
    if ratio_total:
        med = sorted(ratio_medians)[len(ratio_medians) // 2] if ratio_medians else 0.0
        print(f"自称距离 ÷ 真值距离：{ratio_total} 次可比，逐回合中位 {med:.2f}x；"
              f"低估 >= 2 倍的 {under2_total} 次（{100 * under2_total / ratio_total:.0f}%）"
              f"——这个量 <1 说明它永远高估自己的接近程度，于是永远挑小档")
    total_rev = sum(after_total.values())
    if total_rev:
        print(f"反转后有 {total_rev} 次下一步：降档 {after_total.get('downshift', 0)}、"
              f"同档 {after_total.get('same', 0)}、升档 {after_total.get('upshift', 0)}"
              f"（提示词要求反转=走过头后应降档）")
    if quoted_total:
        med = sorted(overshoots)[len(overshoots) // 2]
        print(f"自称像素误差的 move：{quoted_total} 次，其中档位 >= 自身误差 2 倍（一步必然冲过头）的："
              f"{over2_total} 次（{100 * over2_total / quoted_total:.0f}%）；逐回合中位过冲倍数 {med:.2f}x")
    print(f"自相矛盾：文字说“小步/微调”但发 far 共 {mismatch_total.get('text_small_call_far', 0)} 次；"
          f"文字说“大步”却发 fine 共 {mismatch_total.get('text_big_call_fine', 0)} 次")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
