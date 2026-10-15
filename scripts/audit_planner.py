#!/usr/bin/env python3
"""Audit 档 E (planner/executor): what did each layer actually do?

The episode files of a `planner` run hold BOTH layers in one transcript: planner entries
(`assistant` / `result` with no `sub` flag) and every executor entry, flagged `sub` and
tagged with the planner turn that dispatched it.

Questions this answers, all from the transcript alone:

1. How OFTEN does the planner delegate, and does it REWRITE the task between dispatches?
   The design's whole point is "failure -> write the reason into a new task"; a run of
   byte-identical dispatches is the blind-resend failure mode.
2. How much of each sub-budget did the executor use? (A sub that always ends at exactly
   its budget means it never finished; a sub that stops early means it thought it was done.)
3. Did the EXECUTOR get closer to the can than the monolithic `look` arm did? Same model,
   same tools, same world -- only the wrapper differs, so this is the number that says
   whether delegation helped or hurt.
4. Did any gripper `close` land in the graspable band (XY<=22mm and below the can's top)?

USAGE
    python3 scripts/audit_planner.py outputs/episodes/episode_00*_179182*.json
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arm_agent.agent.harness import moved_m  # noqa: E402  (import-path sanity)

TRUTH_PATH = REPO / "outputs" / "init_truth.json"
GRASP_TOL_M = 0.022
CAN_TOP_Z = 0.081  # MEASURED: can body centre 0.0384 + half of 8.5cm


def can_xy(init_idx: int) -> tuple[float, float] | None:
    if not TRUTH_PATH.exists():
        return None
    can = (json.loads(TRUTH_PATH.read_text()).get(str(init_idx)) or {}).get("can")
    return (float(can[0]), float(can[1])) if can else None


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv: list[str]) -> int:
    paths = [Path(p) for a in argv[1:] for p in sorted(glob.glob(a))]
    if not paths:
        print("no episodes given")
        return 1

    print(f"{'episode':<26}{'外层工具':>9}{'dispatch':>9}{'同文重发':>9}{'改写':>6}"
          f"{'子轮/预算':>11}{'子move':>7}{'grip回报':>22}{'XY最近':>8}{'合手z':>7}{'成功':>5}")
    totals = collections.Counter()
    tasks_tot: list[float] = []
    for path in paths:
        data = load(path)
        init_idx = int(path.stem.split("_")[1])
        truth = can_xy(init_idx)
        planner_entries = [e for e in data["transcript"] if not e.get("sub")]
        sub_entries = [e for e in data["transcript"] if e.get("sub")]
        dispatch_entries = [
            e for e in planner_entries
            if str(e.get("result", "")).startswith("[dispatch")
        ]

        # --- did the planner rewrite its task between dispatches? -------------
        tasks: list[str] = []
        budgets: list[int] = []
        for e in dispatch_entries:
            measurement = (e.get("structured_result") or {}).get("measurement") or {}
            tasks.append(str(measurement.get("task_text") or (e.get("structured_result") or {}).get("requested", {}).get("task") or ""))
            budgets.append(int(measurement.get("sub_turns_budget") or 0))
        same = 0
        for a, b in zip(tasks, tasks[1:]):
            if a and b and SequenceMatcher(None, a, b).ratio() > 0.95:
                same += 1
        rewrites = max(0, len(tasks) - 1 - same)
        for a, b in zip(tasks, tasks[1:]):
            if a and b:
                tasks_tot.append(SequenceMatcher(None, a, b).ratio())

        # --- executor effort and effect --------------------------------------
        sub_turns = [int(((e.get("structured_result") or {}).get("measurement") or {}).get("sub_turns") or 0)
                     for e in dispatch_entries]
        verdicts: collections.Counter = collections.Counter()
        closes_z: list[float] = []
        moves = 0
        best = None
        for e in sub_entries:
            sr = e.get("structured_result") or {}
            measurement = sr.get("measurement") or {}
            if sr.get("tool") == "move":
                moves += 1
                eef = measurement.get("eef_end")
                if truth and isinstance(eef, list) and len(eef) == 3:
                    d = ((float(eef[0]) - truth[0]) ** 2 + (float(eef[1]) - truth[1]) ** 2) ** 0.5
                    best = d if best is None else min(best, d)
            if sr.get("tool") == "gripper":
                if (sr.get("requested") or {}).get("action") == "close":
                    z = (measurement.get("eef_end") or [None, None, None])[2]
                    if z is not None:
                        closes_z.append(round(float(z), 3))
                verdicts[str(sr.get("status"))] += 1
        # A close CAN grasp only if it happened below the can's top.
        graspable = sum(1 for z in closes_z if z < CAN_TOP_Z)
        # `verify` entries in outer transcripts carry an empty structured_result
        # (the declare_done branch bypasses the envelope), so .get("tool") is None
        # there -- phrase it as "?" instead of letting sorted() crash on a None key
        # (ep3 of the first high-effort run, 2026-10-15).
        tools = collections.Counter(
            (e.get("structured_result") or {}).get("tool") or "?"
            for e in planner_entries if e.get("result")
        )
        tools.pop("?", None)
        planner_tools = ",".join(f"{k}{v}" for k, v in sorted(tools.items()) if k)
        bud = sum(budgets) or 0
        grip_txt = ",".join(f"{k}×{v}" for k, v in sorted(verdicts.items()))
        print(f"{path.name:<26}{planner_tools:>9}{len(dispatch_entries):>9}{same:>9}{rewrites:>6}"
              f"{f'{sum(sub_turns)}/{bud}':>11}{moves:>7}{grip_txt:>22}"
              f"{(f'{best * 1000:.0f}mm' if best is not None else '—'):>8}"
              f"{(f'{graspable}/{len(closes_z)}' if closes_z else '0/0'):>7}"
              f"{'✓' if data['result'].get('success') else '·':>5}")
        totals["dispatch"] += len(dispatch_entries)
        totals["same"] += same
        totals["sub_turns"] += sum(sub_turns)
        totals["budget"] += bud
        totals["graspable"] += graspable
        totals["closes"] += len(closes_z)
        for k, v in verdicts.items():
            totals[f"grip:{k}"] += v

    print()
    print(f"合计：{len(paths)} 回合、dispatch {totals['dispatch']} 次"
          f"（同文重发 {totals['same']}、改写 {totals['dispatch'] - totals['same'] - len(paths)}）；"
          f"子轮数 {totals['sub_turns']}/{totals['budget']}")
    if tasks_tot:
        print(f"相邻 task 文本相似度 中位 {sorted(tasks_tot)[len(tasks_tot) // 2]:.2f}"
              f"（1.00 = 逐字重发，越低说明改写越多）")
    grip = {k.split(':')[1]: v for k, v in totals.items() if k.startswith('grip:')}
    print(f"执行者夹爪回报合计：{grip}")
    print(f"合手高度低于罐顶(0.081)的次数：{totals['graspable']}/{totals['closes']}"
          f"  ← 只有这一部分才可能真的夹住")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
