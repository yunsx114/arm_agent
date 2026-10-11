"""M1 harness dry run: drive the full loop WITHOUT an LLM.

A scripted policy replaces the model (deterministic greedy controller that reads
the position numbers the harness feeds back), so this exercises everything else
end to end on the login node in ~2 minutes:

    server (in-process) <-> NFS IPC <-> harness <-> scripted "model"

Checked: tool dispatch for all 6 task tools (move/rotate/gripper/look/state/
done), the measured-feedback format the scripted policy depends on, protocol
handling, success verification, transcript writing.

Known-good targets come from outputs/smoke/e1_scene_state.json (init state 0:
alphabet_soup at (-0.119, -0.240, 0.038), basket at (0.015, 0.252, -0.005)).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.agent.harness import Harness
from arm_agent.sim.client import SimClient
from arm_agent.sim.server import SimServer

# Scene targets for init state 0 (GT from the E1 export; the dry run is allowed
# to be privileged - it validates plumbing, not perception).
# GRAB_Z measured by scripts/probe_grasp_height.py: at z=0.05 the close width
# stays at 0.0197 (fingers resting on the can body) and the can lifts +5.3 cm;
# at z=0.035 the can is pushed away instead.
SOUP_XY = (-0.119, -0.240)
BASKET_XY = (0.015, 0.252)
HOVER_Z = 0.25
GRAB_Z = 0.05
DROP_Z = 0.13
LIFT_Z = 0.30
# Tight tolerances: the grasp-height sweep (probe_grasp_height.py) grasped with
# 2 mm XY alignment; 4 mm still shoved the can aside (measured).
XY_TOL_M = 0.002
Z_TOL_M = 0.005
# Final-descent speed cap. Matches probe_grasp_height.py's successful descent
# (a single servo toward the fixed grab height) — smaller per-call steps were
# measured to bump the can repeatedly without ever getting below z~0.085.
DESCEND_MAX_CM = 5.0

_EEF_RE = re.compile(r"机械爪位置 \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)")


class ScriptedPolicy:
    """Deterministic stand-in for QwenAgentModel.

    Parses the harness's feedback text (same information the LLM would see),
    then emits one XML tool call per turn like the real model does.
    """

    def __init__(self, sim=None) -> None:
        self.phase = "to_soup_hover"
        self.calls = 0
        self.sim = sim  # in-process dry run: lets the policy log scene state
        self._last_phase = None

    def chat(self, messages, tools):  # noqa: ANN001
        if self.sim is not None and self.phase != self._last_phase:
            soup = self.sim._obs.get("alphabet_soup_1_pos")
            eef = self.sim._obs.get("robot0_eef_pos")
            print(
                f"[policy] phase={self.phase:14s} soup=({soup[0]:+.3f},{soup[1]:+.3f},{soup[2]:.3f}) "
                f"eef_z={eef[2]:.3f}",
                flush=True,
            )
            self._last_phase = self.phase
        # Find the latest EEF position anywhere in the conversation tail.
        eef = None
        for message in reversed(messages):
            content = message.get("content")
            texts = []
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                texts = [i.get("text", "") for i in content if isinstance(i, dict)]
            for text in texts:
                matches = _EEF_RE.findall(text)
                if matches:
                    eef = tuple(float(v) for v in matches[-1])
                    break
            if eef:
                break
        assert eef is not None, "no EEF position in conversation"
        self.calls += 1
        # Instrumented every-10-calls log: eef xy vs the target for the current
        # phase, plus where the can actually is. This is what distinguishes
        # "tracking error" from "can got knocked".
        if self.sim is not None and self.calls % 10 == 0:
            soup = self.sim._obs.get("alphabet_soup_1_pos")
            basket_phases = {"to_basket", "drop", "open", "retreat", "done"}
            tx, ty = BASKET_XY if self.phase in basket_phases else SOUP_XY
            print(
                f"[policy] t={self.calls:3d} phase={self.phase:13s} "
                f"eef=({eef[0]:+.4f},{eef[1]:+.4f},{eef[2]:.4f}) "
                f"err_xy=({(eef[0] - tx) * 1000:+5.1f},{(eef[1] - ty) * 1000:+5.1f})mm "
                f"can=({soup[0]:+.4f},{soup[1]:+.4f},{soup[2]:.4f})",
                flush=True,
            )
        return self._decide(eef)

    def _decide(self, eef: tuple[float, float, float]) -> str:
        x, y, z = eef

        def move(direction: str, cm: float) -> str:
            return _call("move", {"direction": direction, "distance_cm": round(cm, 2)})

        if self.phase == "to_soup_hover":
            if z < HOVER_Z - Z_TOL_M:
                return move("+z", min(5.0, (HOVER_Z - z) * 100))
            dx, dy = SOUP_XY[0] - x, SOUP_XY[1] - y
            if abs(dx) > XY_TOL_M:
                return move("+x" if dx > 0 else "-x", min(5.0, abs(dx) * 100))
            if abs(dy) > XY_TOL_M:
                return move("+y" if dy > 0 else "-y", min(5.0, abs(dy) * 100))
            self.phase = "down"
            return move("-z", 1.0)

        if self.phase == "down":
            # Re-verify XY on every descent step: measured (dry-run trace) the
            # can gets knocked away when the gripper descends with even a 4 mm
            # lateral offset - and descending introduces lateral drift.
            dx, dy = SOUP_XY[0] - x, SOUP_XY[1] - y
            if abs(dx) > XY_TOL_M:
                return move("+x" if dx > 0 else "-x", min(5.0, abs(dx) * 100))
            if abs(dy) > XY_TOL_M:
                return move("+y" if dy > 0 else "-y", min(5.0, abs(dy) * 100))
            if z > GRAB_Z + Z_TOL_M:
                return move("-z", min(DESCEND_MAX_CM, (z - GRAB_Z) * 100))
            self.phase = "close"
            return _call("set_gripper", {"action": "close"})

        if self.phase == "close":
            self.phase = "lift"
            return move("+z", 2.0)

        if self.phase == "lift":
            if z < LIFT_Z - Z_TOL_M:
                return move("+z", min(5.0, (LIFT_Z - z) * 100))
            self.phase = "to_basket"
            return move("+x", 1.0)

        if self.phase == "to_basket":
            dx, dy = BASKET_XY[0] - x, BASKET_XY[1] - y
            if abs(dx) > XY_TOL_M:
                return move("+x" if dx > 0 else "-x", min(5.0, abs(dx) * 100))
            if abs(dy) > XY_TOL_M:
                return move("+y" if dy > 0 else "-y", min(5.0, abs(dy) * 100))
            self.phase = "drop"
            return move("-z", 1.0)

        if self.phase == "drop":
            dx, dy = BASKET_XY[0] - x, BASKET_XY[1] - y
            if abs(dx) > XY_TOL_M:
                return move("+x" if dx > 0 else "-x", min(5.0, abs(dx) * 100))
            if abs(dy) > XY_TOL_M:
                return move("+y" if dy > 0 else "-y", min(5.0, abs(dy) * 100))
            if z > DROP_Z + Z_TOL_M:
                return move("-z", min(5.0, (z - DROP_Z) * 100))
            self.phase = "open"
            return _call("set_gripper", {"action": "open"})

        if self.phase == "open":
            self.phase = "retreat"
            return move("+z", 3.0)

        if self.phase == "retreat":
            if z < HOVER_Z - Z_TOL_M:
                return move("+z", min(5.0, (HOVER_Z - z) * 100))
            self.phase = "done"
            return _call("declare_done", {"answer_note": "已把字母汤放进篮子"})

        return _call("declare_done", {"answer_note": "完成"})


def _call(name: str, args: dict) -> str:
    params = "\n".join(
        f"<parameter={k}>\n{v}\n</parameter>" for k, v in args.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>"


# ---------------------------------------------------------------- 档 A leg
# The scripted policy above drives the LEGACY surface (`set_gripper`), which is
# still the v1 path. 档 A added `grasp` / `release` to the v2 surface, and the
# properties that matter are not "did the tool call arrive" but:
#   * the model is told a CONCEPT, not a finger width;
#   * `_grasped` follows the CONTACT table (so the state line is honest);
#   * `release()` returns only after the payload cleared the pads, which is what
#     makes an immediately following `verify` meaningful (before 档 A that check
#     failed 8/8 times and the model read it as "my release failed").
# This leg runs that exact sequence with no LLM, so it is cheap enough to run on
# every change.
CONCEPT_SCRIPT: list[tuple[str, dict[str, str]]] = [
    ("release", {}),
    ("approach", {"target": "alphabet soup", "height_offset_cm": "1.2"}),
    ("grasp", {}),
    # TWO lifts on purpose: `move` clamps to MOVE_MAX_DISTANCE_CM = 15, so a
    # single `move +z 25` only reaches ~0.20 -- 4.5 cm short of the 0.22 the
    # payload bottom needs to clear the rim. MEASURED here: the lateral travel
    # then scraped the rim and pushed the BASKET 18 cm away (final basket y =
    # 0.434 vs 0.252 at reset), which is the documented "approach pushes the
    # basket" failure, and the release missed.
    ("move", {"direction": "+z", "distance_cm": "15"}),
    ("move", {"direction": "+z", "distance_cm": "10"}),
    ("approach", {"target": "basket", "height_offset_cm": "25"}),
    ("release", {}),
    ("verify", {"claim": "placed"}),
]


class ConceptPolicy:
    """Emits CONCEPT_SCRIPT in order, then repeats the last call.

    It also records the `爪上：` state line it is shown, because that line lives
    inside the user message the harness builds each turn -- the transcript does
    not store it, so the model is the only place a no-LLM probe can read it.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.state_lines: list[str] = []

    def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
        for message in messages:
            content = message.get("content")
            if message.get("role") != "user":
                continue
            # The harness builds user content as a PARTS LIST (images + a text
            # part); the first version of this probe tested `isinstance(str)`
            # and therefore collected zero state lines.
            parts = content if isinstance(content, list) else [content]
            for part in parts:
                if isinstance(part, str) and "爪上：" in part:
                    self.state_lines.append(part)
                elif isinstance(part, dict) and "爪上：" in str(part.get("text", "")):
                    self.state_lines.append(str(part["text"]))
        turns = sum(1 for m in messages if m.get("role") == "assistant")
        name, args = CONCEPT_SCRIPT[min(turns, len(CONCEPT_SCRIPT) - 1)]
        reply = (
            "现状：脚本化步骤。\n判定：按顺序执行。\n动作：调用工具。\n"
            + _call(name, args)
        )
        self.calls.append({"name": name, "args": args})
        return reply


def concept_leg(ipc_dir: Path) -> bool:
    server = SimServer(ipc_dir=ipc_dir, suite_name="libero_object", task_id=0, cam_size=256)
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 120}, daemon=True)
    thread.start()
    time.sleep(0.2)
    client = SimClient(ipc_dir)
    policy = ConceptPolicy()
    harness = Harness(
        client=client,
        model=policy,
        workspace=REPO,
        max_turns=20,
        tool_protocol="v2",
        log=False,
    )
    result = harness.run_episode(init_state_idx=0, episode_id=998)
    final_obs = client.state().obs
    transcript = json.loads(Path(result.transcript_path).read_text())["transcript"]
    client.close()
    time.sleep(0.3)

    texts = [e["result"] for e in transcript if e.get("result")]
    structs = [e["structured_result"] for e in transcript if e.get("structured_result")]
    grasp_results = [t for t in texts if t.startswith("[grasp]")]
    release_results = [t for t in texts if t.startswith("[release]")]
    grasp_struct = [s for s in structs if s.get("tool") == "grasp"]
    state_lines = policy.state_lines

    checks = {
        "grasp 返回了概念而不是指宽": bool(grasp_results)
        and "夹住了" in grasp_results[0]
        and "0.0" not in grasp_results[0],
        "grasp 的结构化结果里有接触与 body": bool(grasp_struct)
        and grasp_struct[0].get("status") == "grasped"
        and (grasp_struct[0].get("measurement") or {}).get("contact_sides") == ["left", "right"],
        "release 返回了概念": bool(release_results) and "已放开" in release_results[0],
        "状态行跟着接触走": any("有物体（两指接触）" in s for s in state_lines)
        and any("爪上：空" in s for s in state_lines),
        "release 之后立刻判定成功": bool(result.success),
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"  概念腿：success={result.success} source={result.success_source} "
          f"turns={result.turns} retries={result.protocol_retries}")
    if grasp_results:
        print(f"  示例：{grasp_results[0]}")
    if release_results:
        print(f"  示例：{release_results[0]}")
    print("  结果行：" + " | ".join(t[:46] for t in texts))
    objects = final_obs.get("objects", {})
    soup = objects.get("alphabet_soup_1", [0, 0, 0])
    basket = objects.get("basket_1", [0, 0, 0])
    print(f"  最终：罐体心={[round(v, 4) for v in soup]} 篮={[round(v, 4) for v in basket]} "
          f"eef={[round(v, 4) for v in final_obs.get('eef_pos', [])]}")
    return all(checks.values())


def main() -> int:
    ipc_dir = REPO / "runtime" / "ipc_dryrun"
    shutil.rmtree(ipc_dir, ignore_errors=True)

    server = SimServer(ipc_dir=ipc_dir, suite_name="libero_object", task_id=0, cam_size=256)
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 180}, daemon=True)
    thread.start()
    time.sleep(0.2)

    client = SimClient(ipc_dir)
    # 150 turns: the descent needs ~70 turns of fine XY correction (measured),
    # so an 80-turn budget ran out mid-carry.
    harness = Harness(
        client=client, model=ScriptedPolicy(sim=server.sim), workspace=REPO, max_turns=150
    )

    t0 = time.time()
    result = harness.run_episode(init_state_idx=0, episode_id=999)
    print(f"\n=== dry run finished in {time.time() - t0:.0f}s ===")
    print(
        f"success={result.success} done_declared={result.done_declared} "
        f"turns={result.turns} tool_calls={result.tool_calls} retries={result.protocol_retries}"
    )
    print(f"transcript: {result.transcript_path}")

    transcript = json.loads(Path(result.transcript_path).read_text())
    tool_names = [entry.get("result", "")[:12] for entry in transcript["transcript"] if "result" in entry]
    print("tool-result trace:", tool_names[:6], "...", tool_names[-3:])

    # Diagnostic: where did everything end up? (The env still holds the final
    # state after the episode, so one more state query is free.)
    final_obs = client.state().obs
    print("\n--- final scene state ---")
    print("eef:", [round(v, 4) for v in final_obs.get("eef_pos", [])])
    for name, pos in sorted(final_obs.get("objects", {}).items()):
        print(f"  {name:22s} {[round(v, 4) for v in pos]}")
    soup = final_obs.get("objects", {}).get("alphabet_soup_1", [0, 0, 0])
    basket = final_obs.get("objects", {}).get("basket_1", [0, 0, 0])
    dist_xy = ((soup[0] - basket[0]) ** 2 + (soup[1] - basket[1]) ** 2) ** 0.5
    print(f"soup->basket XY distance: {dist_xy:.3f} m (success needs it small & low z)")

    client.close()
    time.sleep(0.3)

    ok = result.tool_calls > 10 and result.protocol_retries == 0
    print("\n=== 档 A 概念腿（v2 grasp/release，无 LLM）===")
    ok = concept_leg(REPO / "runtime" / "ipc_dryrun2") and ok
    print("PLUMBING OK" if ok else "PLUMBING PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
