#!/usr/bin/env python3
"""Capability floor probe: can the model DO align / descend+grasp at all?

The question (user, 2026-10-13): the executor gets stuck circling above the can --
is that a CAPABILITY floor (align + descend + grasp is beyond this model), or some
other factor (task plumbing, step sizes, thinking off)? The planner-era numbers
cannot answer it because everything was measured with confounds: long task texts,
phase boundaries cutting phases mid-air, and (until yesterday's fix) the task text
never even reaching the executor.

So each leg hands the model ONE cleanly-scoped phase task -- now actually delivered
(`run_episode(task_text=...)`, fixed 2026-10-12) -- with a fresh context, and
measures only that phase:

* `align`   : from reset. Task = "move the gripper centre over the blue can, do NOT
              descend/close". Measures visual closed-loop convergence: closest and
              final XY error to the can (ground truth used for SCORING only).
* `descend` : from a SCRIPTED perfect pre-grasp (adapter.align_xy onto the can at
              z_safe, gripper open -- truth used to PLACE, not shown). Task =
              "descend and close". Measures terminal actuation: grasp verdict,
              close heights, whether any close lands under the can top (0.081).

Reading the pair together:
  align fails + descend succeeds  -> the wall is SPATIAL CONVERGENCE, not grasping;
  align succeeds + descend fails  -> the wall is the terminal descent/close;
  both fail                       -> the whole chain is beyond the model as wired.

A `--leg align --ladder` variant swaps `move`'s free distance_cm for the named
rungs (10/1/0.1 cm) -- that separates "cannot decide direction" from "overshoots
every step because the menu is unbounded".

Thinking: `--effort high` turns on strata's reasoning (measured 2026-10-13: works
with tool calls, ~2x wall time, ~390 tok/call incl. reasoning). All prior runs
(every arm ever measured) ran at effort=none, so any leg here is also the first
thinking A/B on this task.

USAGE (sim env; strata must be loaded & healthy):
    bash scripts/simenv.sh scripts/probe_capability_floor.py --legs align,descend \
        --states 0,1,2 --effort none
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

from arm_agent import contracts as C
from arm_agent.agent.harness import SubTaskHarness, moved_m
from arm_agent.agent.strata import StrataAgentModel
from arm_agent.sim.client import SimClient
from arm_agent.sim.server import SimServer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

GRASP_TOL_M = 0.022
CAN_TOP_Z = 0.081  # measured: body centre 0.0384 + 8.5cm/2
HANDOVER_Z = 0.18  # scripted pre-grasp height: ~10 cm above the can top

ALIGN_TASK = (
    "阶段任务：把机械爪的爪心移动到「alphabet soup」蓝罐的正上方停下。"
    "目标罐是蓝色标签的矮胖罐；旁边有橙红/绿红标签的番茄酱矮罐、深绿细颈瓶、红白牛奶纸盒，别看错。"
    "对准标准：全局画面里青色十字压在蓝罐中心，水平偏差 1cm 以内。"
    "只做对准：不要下降、不要合爪。对准后停止移动即可；预算用完控制权会自动交回。"
)
DESCEND_TASK = (
    "阶段任务：夹爪已张开、爪心此刻就在蓝罐（alphabet soup）正上方。"
    "请下降到爪心 z≈0.06~0.07（罐身高度；注意 z≥0.082 已是罐盖沿，在那里合爪只会夹空），"
    "然后调用 close 合爪夹住罐子，夹住后停止动作。预算用完控制权会自动交回。"
)


def can_truth(client: SimClient) -> tuple[float, float, float] | None:
    obs = client.state().obs
    objects = obs.get("objects") or {}
    for key in objects:
        if "soup" in key or "alphabet" in key:
            pos = objects[key]
            return float(pos[0]), float(pos[1]), float(pos[2])
    return None


def run_leg(
    server: SimServer,
    client: SimClient,
    model: StrataAgentModel,
    leg: str,
    state: int,
    ladder: bool,
    budget: int,
) -> dict:
    """One phase-scoped sub-episode on the REAL model. Returns the row to table-ify."""
    for path in sorted(server.ipc_dir.glob(C.IPC_CMD_GLOB)):
        path.unlink(missing_ok=True)

    client.reset(init_state_idx=state)
    can = can_truth(client)
    if can is None:
        raise RuntimeError("can not found in obs.objects -- scoring impossible")

    handover = {"xy_err_mm": None, "z": None}
    if leg == "descend":
        # Script the perfect pre-grasp with GROUND TRUTH (this is the scaffold, not
        # the model's information: the model only ever sees the look surface).
        server.adapter.set_gripper("open")
        out = server.adapter.align_xy((can[0], can[1]), z_safe=HANDOVER_Z)
        end = out.eef_end or server.adapter._eef_now()
        handover = {
            "xy_err_mm": round(float(np.hypot(end[0] - can[0], end[1] - can[1])) * 1000, 1),
            "z": round(float(end[2]), 3),
        }

    harness = SubTaskHarness(
        client=client,
        model=model,
        workspace=REPO,
        max_turns=budget,
        tool_protocol="v2",
        log=True,
        observation_mode="look",
        move_ladder=ladder,
        history_messages=200,
        summary_every=0,
    )
    task = ALIGN_TASK if leg == "align" else DESCEND_TASK
    t0 = time.time()
    result = harness.run_episode(init_state_idx=None, task_text=task)
    elapsed = time.time() - t0

    entries = harness.captured
    moves = [e for e in entries if (e.get("structured_result") or {}).get("tool") == "move"]
    looks = [e for e in entries if (e.get("structured_result") or {}).get("tool") == "observe"]
    grips = [e for e in entries if (e.get("structured_result") or {}).get("tool") == "gripper"]
    verdicts = collections.Counter(
        str((e.get("structured_result") or {}).get("status") or "?") for e in grips
    )
    closes_z = [
        round(float((((e.get("structured_result") or {}).get("measurement") or {}).get("eef_end") or [9, 9, 9])[2]), 3)
        for e in grips
        if (e.get("structured_result") or {}).get("requested", {}).get("action") == "close"
    ]
    # XY error trace over every move (scoring uses truth; the model never sees it).
    xy_err_mm = [
        round(float(np.hypot((m["structured_result"]["measurement"].get("eef_end") or [9, 9, 9])[0] - can[0],
                             (m["structured_result"]["measurement"].get("eef_end") or [9, 9, 9])[1] - can[1])) * 1000, 1)
        for m in moves
        if m.get("structured_result", {}).get("measurement", {}).get("eef_end")
    ]
    final_obs = client.state().obs
    final_err = float(np.hypot(final_obs["eef_pos"][0] - can[0], final_obs["eef_pos"][1] - can[1]))
    closest = min(xy_err_mm) if xy_err_mm else None

    row = {
        "leg": leg,
        "state": state,
        "ladder": ladder,
        "effort": model.reasoning_effort,
        "budget": budget,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "retries": result.protocol_retries,
        "moves": len(moves),
        "looks": len(looks),
        "gripper": dict(verdicts),
        "closes_z": closes_z,
        "closes_below_top": sum(1 for z in closes_z if z < CAN_TOP_Z),
        "grasped": bool(harness._grasped),
        "xy_closest_mm": closest,
        "xy_final_mm": round(final_err * 1000, 1),
        "in_22mm_window": bool(closest is not None and closest <= 22.0),
        "moved_m_total": round(sum(moved_m(m.get("structured_result") or {}) for m in moves), 3),
        "handover": handover,
        "elapsed_s": round(elapsed, 1),
        "error": result.error,
        # The delivered opening is now IN the transcript -- keep the first ~200
        # chars as proof the channel worked for this very leg.
        "opening": str(next((e.get("opening_user", "") for e in entries if e.get("opening_user")), ""))[:200],
    }
    print(
        f"[leg {leg} s{state} ladder={ladder} effort={model.reasoning_effort}] "
        f"turns={result.turns} moves={len(moves)} looks={len(looks)} "
        f"grasped={row['grasped']} closest={closest}mm final={row['xy_final_mm']}mm "
        f"closes_z={closes_z} handover={handover} elapsed={row['elapsed_s']}s",
        flush=True,
    )
    return row


def summarise(rows: list[dict]) -> None:
    def key(r: dict) -> tuple:
        return (r["leg"], r["ladder"], r["effort"])

    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)

    print("\n==== 能力地板汇总（同一组：真值只用于摆位/评分，模型从未见过） ====")
    print(f"{'腿/配置':<34}{'n':>3}{'XY最近mm':>18}{'XY终mm':>16}{'窗口内':>7}{'合爪成功':>9}{'close<顶':>9}{'turns':>8}")
    for (leg, ladder, effort), rs in sorted(groups.items()):
        closest = sorted(r["xy_closest_mm"] for r in rs if r["xy_closest_mm"] is not None)
        finals = sorted(r["xy_final_mm"] for r in rs)
        in_win = sum(1 for r in rs if r["in_22mm_window"])
        grasped = sum(1 for r in rs if r["grasped"])
        below_top = sum(r["closes_below_top"] for r in rs)
        n_closes = sum(len(r["closes_z"]) for r in rs)
        med_c = closest[len(closest) // 2] if closest else None
        med_f = finals[len(finals) // 2] if finals else None
        name = f"{leg}{'+ladder' if ladder else ''} effort={effort}"
        print(
            f"{name:<34}{len(rs):>3}"
            f"{(f'{med_c:.0f}' if med_c is not None else '—'):>18}"
            f"{(f'{med_f:.0f}' if med_f is not None else '—'):>16}"
            f"{in_win}/{len(rs):>4}"
            f"{(f'{grasped}/{len(rs)}' if leg == 'descend' else '—'):>9}"
            f"{(f'{below_top}/{n_closes}' if leg == 'descend' else '—'):>9}"
            f"{sum(r['turns'] for r in rs) / len(rs):>8.0f}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--legs", default="align,descend", help="comma list of align,descend")
    p.add_argument("--states", default="0,1,2")
    p.add_argument("--effort", default="none", choices=["none", "low", "medium", "high"])
    p.add_argument("--ladder", action="store_true", help="named rungs instead of free distance_cm")
    p.add_argument("--align-budget", type=int, default=40)
    p.add_argument("--descend-budget", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--tag", default="", help="suffix for the output json")
    p.add_argument("--keep-ipc", action="store_true")
    args = p.parse_args()

    out_path = REPO / "outputs" / (
        f"capability_floor_{args.effort}{'_ladder' if args.ladder else ''}{args.tag}.json"
    )

    ipc_dir = REPO / "runtime" / "ipc_capfloor"
    if not args.keep_ipc:
        shutil.rmtree(ipc_dir, ignore_errors=True)
    server = SimServer(
        ipc_dir=ipc_dir,
        suite_name="libero_object",
        task_id=0,
        cam_size=256,
        observation_mode="look",  # grid off, truth gated: exactly the executor's world
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 600}, daemon=True)
    thread.start()
    time.sleep(0.3)
    client = SimClient(ipc_dir)
    model = StrataAgentModel(
        base_url="http://127.0.0.1:8080/v1",
        model="strata",
        max_new_tokens=args.max_new_tokens,
        reasoning_effort=args.effort,
    )
    model.load_model()

    legs = [s.strip() for s in args.legs.split(",") if s.strip()]
    states = [int(s) for s in args.states.split(",") if str(s).strip().isdigit()]
    rows: list[dict] = []
    try:
        for state in states:
            for leg in legs:
                budget = args.align_budget if leg == "align" else args.descend_budget
                rows.append(
                    run_leg(server, client, model, leg, state, args.ladder, budget)
                )
    finally:
        client.close()
        time.sleep(0.3)

    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n[saved] {out_path}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
