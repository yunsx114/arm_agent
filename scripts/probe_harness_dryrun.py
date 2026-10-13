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

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent import contracts as C
from arm_agent.agent.harness import Harness, _count_sub_tools
from arm_agent.sim.client import SimClient
from arm_agent.sim.pixel_grounding import (
    AIM_MARK_COLOUR,
    ZOOM_CROP_PX,
    ZOOM_SCALE,
    Camera,
)
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


class PixelPolicy:
    """档 B leg: everything is aimed with PIXELS, never with an object name.

    Privileged on purpose: it computes the pixels by projecting ground-truth
    positions with the same camera maths the server uses -- that is the robot's
    "eyes" for this test, exactly as a model would use its own. What is under test
    is the PLUMBING (grounding, mode filtering, annotation, no GT leak into the
    model's context), not perception quality.

    It also records every model-facing text so the leg can assert that no object
    name and no object coordinate ever reached the model -- the behavioural form
    of the sensor-rights guard (Jev's SensorWorld does the static form by raising
    on forbidden attribute reads).
    """

    OBJECT_TOKENS = ("alphabet_soup", "basket_1", "butter", "cream_cheese", "milk_1",
                     "salad_dressing", "tomato_sauce")

    def __init__(self, server: SimServer, build_script) -> None:
        self.server = server
        self.build_script = build_script
        self.script: list[tuple[str, dict[str, Any]]] = []
        self.sent: list[str] = []
        self.calls: list[str] = []
        self.frames = 0
        self.images_annotated = False
        self.calls_made = 0
        # Frames that came back carrying an `aim` ring (see the pixel scan below).
        self.ring_frames = 0
        # Magnified crops, keyed by the zoom key the server sends them under.
        self.zoom_frames = 0

    def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
        if not self.script:
            self.script = self.build_script(self.server)
        # Record only what comes AFTER the first assistant turn: the system prompt
        # and the opening task message are not tool feedback, and the task string
        # itself names the object (LIBERO: "pick up the alphabet soup ..."), so
        # scanning them would report the GOAL as a leak.
        if sum(1 for m in messages if m.get("role") == "assistant") > 0:
            for message in messages:
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                parts = content if isinstance(content, list) else [content]
                for part in parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        self.sent.append(str(part.get("text", "")))
                    elif isinstance(part, dict) and part.get("type") == "image":
                        self.frames += 1
                        # The magnifier is attached as a THIRD image, on the turns that
                        # carried a pixel tool. Its size is what identifies it: 64 px at
                        # 4x (see pixel_grounding.ZOOM_CROP_PX / ZOOM_SCALE), vs the
                        # 256 px cameras.
                        if np.asarray(part["image"]).shape[0] == ZOOM_CROP_PX * ZOOM_SCALE:
                            self.zoom_frames += 1
                        # Annotation is drawn server-side, so the only honest check
                        # is on the PIXELS the model receives: one of the grid greys
                        # has to be present in a frame that is otherwise a normal
                        # render. (Brighter lines + labels every 32 px, dim minor
                        # lines every 8 px -- see pixel_grounding.annotate_frame.)
                        arr = np.asarray(part["image"]).reshape(-1, 3).astype(int)
                        for grey in ((96, 106, 116), (150, 158, 164)):
                            hits = (np.abs(arr - np.array(grey)).sum(axis=1) <= 6).sum()
                            if int(hits) > 200:
                                self.images_annotated = True
                        # The `aim` ring has to be on the PIXELS, not just in the
                        # return value: the whole point of the tool is that the model
                        # compares the ring against the object in the frame it gets
                        # back. AIM_MARK_COLOUR with a generous tolerance (the PNG round
                        # trip and the halo antialiasing both shift the value).
                        ring = (np.abs(arr - np.array(AIM_MARK_COLOUR)).sum(axis=1) <= 60).sum()
                        if int(ring) >= 10:
                            self.ring_frames += 1
                    elif isinstance(part, str):
                        self.sent.append(part)
        # Advance on CALL COUNT, not on the number of assistant messages in the
        # history: `compact_history` drops old turns once the window fills, so
        # counting messages made this policy saturate on one script entry forever
        # (measured: 19 identical `target` calls at the end of a 32-turn leg, which
        # looked exactly like a tool that could not finish).
        index = min(self.calls_made, len(self.script) - 1)
        self.calls_made += 1
        name, args = self.script[index]
        self.calls.append(name)
        return "现状：脚本化像素步骤。\n判定：按顺序执行。\n动作：调用工具。\n" + _call(name, args)


def pixel_script(server: SimServer) -> list[tuple[str, dict[str, Any]]]:
    """The pixel-only version of the task, built from where the objects are."""
    env = server.sim.env.env
    objects = server.sim.pack(server.sim._obs)["objects"]
    agent = Camera.from_env(env, "agentview", 256, 256)

    def px(name: str) -> tuple[int, int]:
        u, v, _ = agent.project(np.asarray(objects[name], dtype=float))
        return int(round(u)), int(round(v))

    can_u, can_v = px("alphabet_soup_1")
    basket_u, basket_v = px("basket_1")
    return [
        ("release", {}),
        # `approach` must be refused in this mode. Asked EARLY on purpose: the
        # refusal costs one protocol retry and executes nothing, so the task can
        # still be finished afterwards (asked at the end it never runs, because
        # the episode stops the moment the environment reports success).
        ("approach", {"target": "basket", "height_offset_cm": "25"}),
        # `aim` is the dry run: it must mark the frame and move NOTHING.
        ("aim", {"camera": "agent", "u": can_u, "v": can_v}),
        # ...it is also the only pixel tool allowed on the wrist view...
        ("aim", {"camera": "wrist", "u": 128, "v": 128}),
        # ...a pixel outside the frame is refused WITH A REASON (never clamped)...
        ("aim", {"camera": "agent", "u": -4, "v": 900}),
        # ...and MOVING to a wrist pixel is refused as a wrong-tool error.
        ("target", {"camera": "wrist", "u": can_u, "v": can_v}),
        # hover 12 cm above the can's visible surface, then 1.2 cm above it, i.e.
        # the two-stage aim the pixel granularity forces (3.7 mm/px up here).
        # Repeated because ONE CALL IS A SHORT BLOCK (<= 8 corrections, and each
        # correction is a `move` capped at 5 cm); a repeat on a point that is
        # already reached costs nothing and answers `arrived`.
        ("target", {"camera": "agent", "u": can_u, "v": can_v,
                    "offset_mm": [0, 0, 120]}),
        ("target", {"camera": "agent", "u": can_u, "v": can_v,
                    "offset_mm": [0, 0, 120]}),
        ("target", {"camera": "agent", "u": can_u, "v": can_v,
                    "offset_mm": [0, 0, 12]}),
        ("target", {"camera": "agent", "u": can_u, "v": can_v,
                    "offset_mm": [0, 0, 12]}),
        ("grasp", {}),
        ("move", {"direction": "+z", "distance_cm": "15"}),
        ("move", {"direction": "+z", "distance_cm": "10"}),
        # 19 cm above the basket's visible floor: the payload's bottom then sits
        # ~14 cm above the rim (rim 0.1513; payload bottom = EEF - 0.050).
        ("target", {"camera": "agent", "u": basket_u, "v": basket_v,
                    "offset_mm": [0, 0, 190]}),
        ("target", {"camera": "agent", "u": basket_u, "v": basket_v,
                    "offset_mm": [0, 0, 190]}),
        ("target", {"camera": "agent", "u": basket_u, "v": basket_v,
                    "offset_mm": [0, 0, 190]}),
        ("release", {}),
    ]


def pixel_leg(ipc_dir: Path) -> bool:
    server = SimServer(
        ipc_dir=ipc_dir,
        suite_name="libero_object",
        task_id=0,
        cam_size=256,
        observation_mode="pixels",
        pixel_grid_px=32,
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 120}, daemon=True)
    thread.start()
    time.sleep(0.2)
    client = SimClient(ipc_dir)
    policy = PixelPolicy(server=server, build_script=pixel_script)
    harness = Harness(
        client=client,
        model=policy,
        workspace=REPO,
        max_turns=32,
        tool_protocol="v2",
        log=False,
        observation_mode="pixels",
    )
    advertised = [s["function"]["name"] for s in harness.tool_schemas]
    result = harness.run_episode(init_state_idx=0, episode_id=997)
    transcript = json.loads(Path(result.transcript_path).read_text())["transcript"]
    client.close()
    time.sleep(0.3)

    texts = [e["result"] for e in transcript if e.get("result")]
    aim_results = [t for t in texts if t.startswith("[aim")]
    target_results = [t for t in texts if t.startswith("[target")]
    # Every turn's structured envelope, for the assertions that need NUMBERS
    # (did `aim` move the arm?) rather than the model-facing sentence.
    envelopes = [e.get("structured_result") or {} for e in transcript if e.get("structured_result")]
    aim_envelopes = [s for s in envelopes if s.get("tool") == "aim"]
    target_envelopes = [s for s in envelopes if (s.get("tool") or "").endswith("target")]
    aim_moved = [
        s for s in aim_envelopes
        if (s.get("measurement") or {}).get("eef_start")
        != (s.get("measurement") or {}).get("eef_end")
        or any(abs(v) > 1e-9 for v in ((s.get("measurement") or {}).get("measured_delta_m") or [0, 0, 0]))
    ]
    # Only TOOL RESULTS and state lines are scanned. The initial user message
    # legitimately quotes the task (LIBERO's own task string is "pick up the
    # alphabet soup and place it in the basket"), so a bare name search over the
    # whole context would "find" the goal statement and cry wolf.
    toolish = [t for t in policy.sent if t.startswith("[") or "爪上：" in t]
    joined = "\n".join(toolish)
    leaked_names = [tok for tok in PixelPolicy.OBJECT_TOKENS if tok in joined]
    leaked_coords = re.findall(r"\(\s*[-+]?\d\.\d{3}\s*,\s*[-+]?\d\.\d{3}\s*,\s*[-+]?\d\.\d{3}\)", joined)
    # 档 B v2: the model must not receive the WORLD POINT it just asked for
    # (`surface_xyz` / `target_xyz` / `offset_m`) or the marker echo. Those stay in
    # the transcript; the model is supposed to decide from the ring in the picture.
    leaked_keys = [key for key in ("surface_xyz", "target_xyz", "offset_m", "depth_m") if f'"{key}"' in joined]
    short_blocks = [
        s for s in target_envelopes
        if (s.get("measurement") or {}).get("iterations") is not None
    ]
    bounded = [s for s in short_blocks if int((s.get("measurement") or {})["iterations"]) <= 8]
    checks = {
        "工具面里没有 approach，但有 aim/target": "approach" not in advertised
        and "aim" in advertised and "target" in advertised,
        "aim 只标注不动手": len(aim_results) >= 3 and not aim_moved,
        "aim 可用腕部相机，target 不行": any("wrist" in t for t in aim_results)
        and not any("wrist" in t for t in target_results)
        and any(e.get("protocol_retry") for e in transcript),
        "越界像素被拒且给了理由": any("拒绝" in t for t in aim_results),
        "target 走了多次且给出了概念": len(target_results) >= 4
        and all("已到位" in t or "没到位" in t or "被挡住" in t or "拒绝" in t for t in target_results),
        "短动作块有上限（每次 <= 8 个修正步）": bool(short_blocks) and len(bounded) == len(short_blocks),
        "工具结果/状态行里没有物体名": not leaked_names,
        "工具结果里没有物体坐标（只有爪心）": not leaked_coords,
        "不给模型世界坐标/深度键": not leaked_keys,
        "approach 在 pixels 模式被拒且给了替代": any(
            e.get("protocol_retry") for e in transcript
        ),
        "任务由 target 完成": bool(result.success),
        "画面带网格标注": bool(policy.images_annotated),
        "红圈画在了返回的画面上": policy.ring_frames >= 1,
        "每张 aim/target 都附了放大图": policy.zoom_frames >= len(aim_results) + len(target_results),
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"  像素腿：success={result.success} source={result.success_source} "
          f"turns={result.turns} retries={result.protocol_retries} error={result.error!r}")
    print(f"  脚本调用: {policy.calls}")
    if target_results:
        print(f"  示例：{target_results[0]} / {target_results[-1]}")
    if leaked_names:
        print(f"  泄漏的物体名: {leaked_names}")
        for text in toolish:
            if any(tok in text for tok in leaked_names):
                print(f"    命中：{text[:160]!r}")
                break
    if leaked_coords:
        print(f"  泄漏的坐标: {leaked_coords[:4]}")
    return all(checks.values())


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
          f"turns={result.turns} retries={result.protocol_retries} error={result.error!r}")
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


def raw_next_call(server: SimServer, ladder: bool = False) -> tuple[str, dict[str, Any]]:
    """The next `move`/`gripper` call a controller would make, from TRUTH.

    Privileged on purpose (same as the other legs): this answers "are two tools enough
    to do the task at all", which is a plumbing claim. Whether a MODEL can do it with
    only those two tools is the experiment this leg makes possible, not the thing it
    measures.

    The shape of the controller is also the recipe the `raw` prompt teaches: one small
    step, then look again (the state is re-read from the simulator on every turn).

    With `ladder=True` the SAME controller emits `size` instead of `distance_cm`, picking
    the rung from the remaining error (>3cm -> far, >3mm -> near, else fine). That is the
    model's whole job under 档 D, so if the task still completes here, the ladder is
    sufficient for the task and the only open question is whether the model picks rungs
    sensibly.
    """
    obs = server.sim._obs
    eef = np.asarray(obs[C.OBS_EEF_POS], dtype=float)
    objects = server.sim.pack(obs)["objects"]
    can = np.asarray(objects["alphabet_soup_1"], dtype=float)
    basket = np.asarray(objects["basket_1"], dtype=float)
    width = float(np.abs(np.asarray(obs[C.OBS_GRIPPER_QPOS])).mean())
    carried, _ = server.adapter.held_object()

    def step_cm(cm: float) -> dict[str, Any]:
        """The distance argument for a DELIBERATE step of intent `cm`.

        A caller that has already decided "raise by 5cm" gets the rung that is closest
        without exceeding it: 5cm -> `near` (1cm), repeated by the loop.
        """
        if not ladder:
            return {"distance_cm": f"{cm:.1f}"}
        if cm >= 10.0:
            return {"size": "far"}
        if cm >= 1.0:
            return {"size": "near"}
        return {"size": "fine"}

    def toward(target: np.ndarray, tol: float) -> tuple[str, dict[str, Any]] | None:
        dx, dy = target[0] - eef[0], target[1] - eef[1]
        if abs(dx) > tol or abs(dy) > tol:
            axis, delta = ("x", dx) if abs(dx) >= abs(dy) else ("y", dy)
            error_cm = abs(delta) * 100.0
            if not ladder:
                step = min(5.0, max(1.0, error_cm))
                return "move", {"direction": f"{'+' if delta > 0 else '-'}{axis}",
                                "distance_cm": f"{step:.1f}"}
            # Largest rung that does NOT overshoot the remaining error. A controller that
            # picked `far` for a 4cm error would orbit forever (measured while writing
            # this leg: 64 consecutive `far` steps, 0 far from arriving) -- the rung must
            # be chosen from the error, which is exactly the judgement the model has to
            # make and exactly what this leg is not allowed to get wrong.
            if error_cm >= 10.0:
                rung = "far"
            elif error_cm >= 1.0:
                rung = "near"
            else:
                rung = "fine"
            return "move", {"direction": f"{'+' if delta > 0 else '-'}{axis}", "size": rung}
        return None

    def rung_for_z(error_cm: float, *, up: bool) -> dict[str, Any]:
        """The rung for a VERTICAL step.

        Asymmetric on purpose, and the asymmetry is a MEASURED property of the scene, not
        a preference: raising is free (nothing is in the way), while descending with the
        coarse rung drove the gripper 9.7cm in one call -- straight through the 3cm margin
        the close-on-the-can rule is built around. So `up` may use the whole ladder and
        `down` is capped at `near`. A 1cm descent costs turns; a 10cm one costs the can.
        """
        if up:
            return step_cm(error_cm)
        if ladder:
            return {"size": "fine" if error_cm < 1.0 else "near"}
        # Clamped to 0.5cm: MEASURED failure from an unclamped version -- an error of
        # 0.04cm formatted as "0.0", which the protocol correctly rejects as out of range
        # (move takes 0.1~15), so the same rejected call repeated until the turn cap
        # (27 retries, episode lost). Under the ladder this cannot happen at all: the
        # model picks a WORD and the code owns the number.
        return {"distance_cm": f"{max(0.5, min(3.0, error_cm)):.1f}"}

    # ORDER MATTERS, and getting it wrong produced a z-oscillation (measured while
    # building this leg): a blanket "keep z above the travel height" rule checked
    # BEFORE the descent rule fights the descent forever -- `-z 3cm` down to 0.214,
    # `+z 5cm` back up to 0.263, repeat. Raise only while travelling, descend only
    # once aligned.
    if width < 0.030:  # pads are half closed at reset: open first (the 档 A lesson)
        return "gripper", {"action": "open"}
    if not carried:
        step = toward(can, tol=0.012)
        if step:
            if eef[2] < 0.24:  # travel height before moving sideways
                return "move", {"direction": "+z", **rung_for_z(5.0, up=True)}
            return step
        if eef[2] > can[2] + 0.030:
            return "move", {"direction": "-z", **rung_for_z((eef[2] - can[2]) * 100.0, up=False)}
        return "gripper", {"action": "close"}
    # carrying: up, across, down, open, and let the ENV judge.
    # 2.2cm for the drop, not 1.5cm: the ladder's 0.92cm `near` step couples ~1cm of
    # off-axis drift per call (measured), so a 1.5cm window could not be settled -- the
    # scripted run ping-ponged for 99 `near` calls and ran out of turns. 2.2cm is the
    # MEASURED grasp tolerance of this scene, i.e. an honest allowance rather than a
    # number bent to make the leg pass.
    step = toward(basket, tol=0.022)
    if step:
        if eef[2] < 0.25:
            return "move", {"direction": "+z", **rung_for_z(10.0, up=True)}
        return step
    if eef[2] > 0.20:
        return "move", {"direction": "-z", **rung_for_z((eef[2] - 0.20) * 100.0, up=False)}
    return "gripper", {"action": "open"}


class RawPolicy:
    """Scripted policy for the `raw` leg: the two-tool task, no localisation aid.

    The first three calls deliberately ASK FOR the tools this mode removed. A refusal
    that executes nothing costs one protocol retry and is the behavioural proof that
    the surface is enforced rather than merely unadvertised (the same trick the pixel
    leg uses on `approach`).
    """

    PRELUDE = (
        ("aim", {"u": "64", "v": "128"}),
        ("target", {"u": "64", "v": "128"}),
        ("approach", {"target": "alphabet soup"}),
    )
    # Extra calls emitted BEFORE the geometry loop: the ladder leg puts its rung
    # calibration here. `SCRIPT` is what the loop below walks.
    CALIBRATION: tuple = ()
    LADDER = False

    def __init__(self, server: SimServer) -> None:
        self.server = server
        self.sent: list[str] = []
        self.calls: list[str] = []
        self.grid_frames = 0
        self.frames = 0
        self.opens = 0
        self.done = False
        self.system = ""

    @property
    def SCRIPT(self) -> tuple:  # noqa: N802 - mirrors the class-level tuple
        return self.PRELUDE + self.CALIBRATION

    def _next_call(self) -> tuple[str, dict[str, Any]]:
        """The next call. Overridden by `look`'s policy to insert looks."""
        return raw_next_call(self.server, ladder=self.LADDER)

    def _capture(self, messages) -> None:  # noqa: ANN001
        """(unused hook kept for subclasses that need the raw message list)"""
        del messages

    def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
        for message in messages:
            if message.get("role") == "system":
                self.system = str(message.get("content", ""))
        if sum(1 for m in messages if m.get("role") == "assistant") > 0:
            for message in messages:
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                for part in content if isinstance(content, list) else [content]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        self.sent.append(str(part.get("text", "")))
                    elif isinstance(part, dict) and part.get("type") == "image":
                        self.frames += 1
                        arr = np.asarray(part["image"]).reshape(-1, 3).astype(int)
                        for grey in ((96, 106, 116), (150, 158, 164)):
                            if int((np.abs(arr - np.array(grey)).sum(axis=1) <= 6).sum()) > 200:
                                self.grid_frames += 1
                    elif isinstance(part, str):
                        self.sent.append(part)
        if len(self.calls) < len(self.SCRIPT):
            name, args = self.SCRIPT[len(self.calls)]
            self.calls.append(name)
            return "现状：脚本化步骤。\n判定：试探被禁工具。\n动作：调用工具。\n" + _call(name, args)
        if self.done:
            name, args = "verify", {"claim": "placed"}
        else:
            name, args = self._next_call()
            if name == "gripper" and args["action"] == "open":
                self.opens += 1
                # open(pads) -> carry -> open(over the basket): the third open is the
                # release, after which asking the ENV is the only sane move left.
                if self.opens >= 3:
                    self.done = True
        self.calls.append(name)
        return "现状：脚本化步骤。\n判定：按几何驱动。\n动作：调用工具。\n" + _call(name, args)


def raw_leg(ipc_dir: Path) -> bool:
    """Minimum surface: move + gripper + verify, no pixel tools, no grid."""
    server = SimServer(
        ipc_dir=ipc_dir,
        suite_name="libero_object",
        task_id=0,
        cam_size=256,
        observation_mode="raw",
        pixel_grid_px=32,
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 120}, daemon=True)
    thread.start()
    time.sleep(0.2)
    client = SimClient(ipc_dir)
    policy = RawPolicy(server=server)
    harness = Harness(
        client=client,
        model=policy,
        workspace=REPO,
        max_turns=60,
        tool_protocol="v2",
        log=False,
        observation_mode="raw",
    )
    advertised = sorted(s["function"]["name"] for s in harness.tool_schemas)
    result = harness.run_episode(init_state_idx=0, episode_id=995)
    transcript = json.loads(Path(result.transcript_path).read_text())["transcript"]
    client.close()
    time.sleep(0.3)
    texts = [e["result"] for e in transcript if e.get("result")]
    grip_results = [t for t in texts if t.startswith("[gripper")]
    closed = [t for t in grip_results if "close" in t]
    toolish = [t for t in policy.sent if t.startswith("[") or "爪上：" in t]
    joined = "\n".join(toolish)
    checks = {
        "工具面正好是 move/gripper/verify": advertised == ["gripper", "move", "verify"],
        "没有像素工具、也没有 approach": all(
            name not in advertised for name in ("aim", "target", "approach", "grasp", "release")
        ),
        "aim/target/approach 被拒（各一次协议重试）": sum(
            1 for e in transcript if e.get("protocol_retry")
        ) >= 3,
        "夹爪只回报概念，不给指宽": bool(closed) and all(
            "夹住" in t or "夹空" in t or "接触不稳" in t for t in closed
        ) and "width 0." not in joined,
        # Two cameras per turn that actually RAN a tool (a refused call produces no
        # frames at all -- the retry happens before dispatch), so the budget is
        # turns minus retries, not turns.
        "每轮都带两张画面": policy.frames >= 2 * (result.turns - result.protocol_retries),
        "raw 模式的画面不带网格": policy.grid_frames == 0 and policy.frames > 0,
        "两个工具就能完成任务": bool(result.success),
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"  raw 腿：success={result.success} source={result.success_source} "
          f"turns={result.turns} retries={result.protocol_retries} error={result.error!r}")
    print(f"  工具面：{advertised}")
    print(f"  脚本调用：{policy.calls}")
    if closed:
        print(f"  示例：{closed[0]} / {closed[-1]}")
    return all(checks.values())


def ladder_leg(ipc_dir: Path) -> bool:
    """档 D: `move` takes a NAMED rung (far/near/fine = 10cm/1cm/1mm) instead of a distance.

    Three things are asserted, and each one is the reason the ladder exists:

    1. The advertised `move` schema has `size` and NOT `distance_cm`, and the system
       prompt carries the upshift/downshift rule.
    2. A call that still tries to pass a distance is REFUSED, not silently converted --
       otherwise the model could keep doing the arithmetic it demonstrably gets wrong
       (raw mode: 14cm y error) and the ladder would hide it.
    3. The rungs mean what they say. A calibration of three explicit rung moves measures
       ~10cm / ~1cm / ~1mm, and the whole task still completes with every move coming
       from the three rungs.
    """
    server = SimServer(
        ipc_dir=ipc_dir,
        suite_name="libero_object",
        task_id=0,
        cam_size=256,
        observation_mode="raw",
        pixel_grid_px=32,
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 120}, daemon=True)
    thread.start()
    time.sleep(0.2)
    client = SimClient(ipc_dir)
    policy = LadderPolicy(server=server)
    harness = Harness(
        client=client,
        model=policy,
        workspace=REPO,
        # The ladder costs TURNS: a 1cm rung cannot cross the ~25cm to the basket in one
        # call, and each coarse step couples ~1.2cm of off-axis drift (measured) that the
        # next step has to pay back. 70 turns ran out mid-carry; the real question for a
        # model run is whether MAXTURNS=80 is enough, which this leg deliberately leaves
        # unanswered -- it measures sufficiency, not budget.
        max_turns=120,
        tool_protocol="v2",
        log=False,
        observation_mode="raw",
        move_ladder=True,
    )
    move_schema = next(s["function"] for s in harness.tool_schemas if s["function"]["name"] == "move")
    advertised_props = sorted(move_schema["parameters"]["properties"])
    result = harness.run_episode(init_state_idx=0, episode_id=996)
    transcript = json.loads(Path(result.transcript_path).read_text())["transcript"]
    final_obs = client.state().obs
    client.close()
    time.sleep(0.3)

    texts = [e["result"] for e in transcript if e.get("result")]
    # A refused call never dispatches, so the ONLY place the refusal exists is the retry
    # note the harness hands back to the model -- which is what `policy.sent` collected.
    joined = "\n".join(policy.sent)
    # Per executed move: which rung, what was requested, what actually happened.
    moves = []
    for entry in transcript:
        sr = entry.get("structured_result") or {}
        if sr.get("tool") != "move":
            continue
        measurement = sr.get("measurement") or {}
        requested = sr.get("requested") or {}
        delta = measurement.get("measured_delta_m") or [0, 0, 0]
        # Measure ON THE COMMANDED AXIS. MEASURED while writing this leg: a `+x far`
        # step reports (-0.0089, 0.0006, -0.0118) -- the OSC couples a 1.2cm z excursion
        # into a horizontal move, so "the biggest component" would score a 1cm `near`
        # step as 1.31cm and a 1mm `fine` step as whatever the wobble happened to be.
        # The rung's contract is about the axis it names.
        axis = {"x": 0, "y": 1, "z": 2}.get(str(requested.get("direction", ""))[-1], 0)
        moves.append({
            "rung": measurement.get("move_size"),
            "requested_cm": float(requested.get("distance_cm") or 0.0),
            "moved_cm": abs(float(delta[axis])) * 100.0,
            "wobble_cm": max(abs(float(v)) for v in delta) * 100.0,
        })
    # Calibration = the first four moves that actually RAN (a refused call produces none),
    # in order: +x far, -x near, -x fine, then the gap_px call.
    calibration = moves[:4]
    by_rung = {m["rung"]: m for m in calibration}
    nominal = {"far": 10.0, "near": 1.0, "fine": 0.1}
    # Tolerance per rung: 25% or 0.5mm, whichever is larger -- a 1mm rung measured
    # through a 1kHz OSC servo is not going to be 1.000mm.
    rung_ok = {}
    for rung, want in nominal.items():
        got = by_rung.get(rung, {}).get("moved_cm")
        rung_ok[rung] = got is not None and abs(got - want) <= max(0.25 * want, 0.05)
    requested_values = {round(m["requested_cm"], 3) for m in moves}
    checks = {
        "move 的 schema 有 gap_px+size、没有 distance_cm": (
            advertised_props == ["direction", "gap_px", "size"]
        ),
        "档位值与像素差都写进了参数描述": (
            "far=10厘米" in move_schema["parameters"]["properties"]["size"]["description"]
            and "像素" in move_schema["parameters"]["properties"]["gap_px"]["description"]
        ),
        "提示词叫它报像素、别自己换算": (
            "gap_px" in policy.system and "不要自己换算成厘米" in policy.system
        ),
        "raw 的 1~5cm 配方被换成了像素档位表": (
            "一次只走一小段（1~5cm）" not in policy.system
            and "≥ 32 像素" in policy.system
            and "不要自己把像素换算成厘米" in policy.system
        ),
        "仍然传 distance_cm 的调用被拒（不静默换算）": "ladder_rejects" in joined,
        "乱编的档位名被拒": "invalid_enum" in joined and "size=huge" in joined,
        "被拒的调用没有动机械爪（标定就是头四次 move）": [
            m["rung"] for m in calibration[:3]
        ] == ["far", "near", "fine"],
        "三个档位实测量≈10cm/1cm/1mm": all(rung_ok.values()),
        "gap_px=80 被代码换算成 far（约 30cm 的真差）": (
            len(calibration) > 3 and calibration[3]["rung"] == "far"
            and abs(calibration[3]["requested_cm"] - 10.0) < 0.01
        ),
        "所有实际执行的 move 都来自三档之一": requested_values <= {10.0, 1.0, 0.1} and bool(moves),
        "结果里带上了档位名与实测位移": all(m["rung"] in nominal for m in moves),
        "raw 模式画面仍然不带网格": policy.grid_frames == 0 and policy.frames > 0,
        "阶梯下两工具仍能完成任务": bool(result.success),
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"  阶梯腿：success={result.success} source={result.success_source} "
          f"turns={result.turns} retries={result.protocol_retries} error={result.error!r}")
    print(f"  标定：{ {r: round(by_rung.get(r, {}).get('moved_cm', -1), 2) for r in nominal} }")
    print(f"  执行的 move {len(moves)} 次，档位分布："
          f"{ {r: sum(1 for m in moves if m['rung'] == r) for r in nominal} }")
    print(f"  重试提示：{[t for t in policy.sent if 'ladder_rejects' in t or 'invalid_enum' in t][:2]}")
    print(f"  终局：can={[round(float(v), 4) for v in (final_obs.get('objects') or {}).get('alphabet_soup_1', [])]} "
          f"basket={[round(float(v), 4) for v in (final_obs.get('objects') or {}).get('basket_1', [])]}")
    if texts:
        print(f"  示例：{texts[0][:80]}")
    return all(checks.values())


class LadderPolicy(RawPolicy):
    """档 D: the same task, but every move must name a rung.

    The prelude changes meaning here: instead of tools this mode removed, it asks for
    the vocabulary the ladder removed -- a `move` carrying `distance_cm` (the exact
    arithmetic the ladder replaces) and a `move` with an invented rung. Both must be
    refused, because the ladder's value depends on the model NOT being able to fall
    back to numbers it gets wrong.

    Then three calibration calls, one per rung, in +x and back. They are what turns
    "the enum exists" into "the rung is 10cm / 1cm / 1mm" -- the claim the model is
    being asked to reason with.
    """

    LADDER = True
    PRELUDE = (
        ("move", {"direction": "+x", "distance_cm": "5"}),
        ("move", {"direction": "+x", "size": "huge"}),
        ("aim", {"u": "64", "v": "128"}),
    )
    CALIBRATION = (
        ("move", {"direction": "+x", "size": "far"}),
        ("move", {"direction": "-x", "size": "near"}),
        ("move", {"direction": "-x", "size": "fine"}),
        # A PIXEL GAP, not a rung: the code must convert it with the camera scale and snap
        # it. 80px at ~3.7mm/px is ~30cm, i.e. `far`. This is the input the model is
        # measurably good at producing (its quoted gap runs 1.2-3.2x of truth) and the
        # conversion is the one it fails at (~11x).
        ("move", {"direction": "-x", "gap_px": "80"}),
    )


class LookPolicy(RawPolicy):
    """`look` mode: the same minimal surface, but SEEING is an action.

    Three things this leg has to prove, because each is the reason the mode exists:

    1. The task message carries NO frame -- the model opens blind and must decide to look.
    2. Frames arrive ONLY as the result of a `look` call (never attached to a turn that ran
       `move`/`gripper`).
    3. No frame is ever elided: the number of images in the prompt KEEPS GROWING with the
       number of looks. With the old `image_keep=2` budget it would plateau at 2, which is
       exactly why "did my last step take me past it?" was unanswerable.
    """

    LADDER = False
    PRELUDE = (
        ("aim", {"u": "64", "v": "128"}),          # banned here: no pixel tools
        ("look", {"camera": "agent"}),             # the only legal way to see
    )

    def __init__(self, server: SimServer) -> None:
        super().__init__(server)
        self.looks = 0
        self.frames_before_look = 0
        self.max_images_seen = 0
        self.last_seen_images = 0
        self.image_series: list[int] = []
        self.initial_message_images = -1

    def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
        total_images = sum(
            sum(1 for it in (m.get("content") or []) if isinstance(it, dict) and it.get("type") == "image")
            for m in messages
            if isinstance(m.get("content"), list)
        )
        self.max_images_seen = max(self.max_images_seen, total_images)
        self.last_seen_images = total_images
        self.image_series.append(total_images)
        if self.initial_message_images < 0:
            self.initial_message_images = total_images
        # "Before the first look" means before a `look` CALL was ever emitted -- whoever
        # emitted it (the script's prelude counts too). Using the subclass's own counter
        # here flagged the frames from the scripted look as a violation.
        if self.calls.count("look") == 0:
            for message in messages[1:]:
                content = message.get("content")
                if isinstance(content, list) and any(
                    isinstance(it, dict) and it.get("type") == "image" for it in content
                ):
                    self.frames_before_look += 1
        return super().chat(messages, tools=tools, **kwargs)

    def _next_call(self) -> tuple[str, dict[str, Any]]:
        """Geometry, but stop and LOOK every few steps like the prompt teaches."""
        if self.calls.count("look") < 3 and len(self.calls) % 9 == 0:
            self.looks += 1
            return "look", {"camera": "wrist" if self.calls.count("look") % 2 else "agent"}
        return raw_next_call(self.server, ladder=False)


def look_leg(ipc_dir: Path) -> bool:
    """档 E: `move` + `gripper` + `look` + `verify`, no frames unless the model looks."""
    server = SimServer(
        ipc_dir=ipc_dir,
        suite_name="libero_object",
        task_id=0,
        cam_size=256,
        observation_mode="look",
        pixel_grid_px=32,
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 120}, daemon=True)
    thread.start()
    time.sleep(0.2)
    client = SimClient(ipc_dir)
    policy = LookPolicy(server=server)
    harness = Harness(
        client=client,
        model=policy,
        workspace=REPO,
        max_turns=70,
        tool_protocol="v2",
        log=False,
        observation_mode="look",
        # The MESSAGE cap also removes images (it removes whole messages), so a
        # 32-message window erases the frames a walk-and-look policy needs: MEASURED in
        # this leg, the image count per prompt went 0,2,2,...,2,0,0,0 -- the last three
        # looks had fallen out of the tail. With the cap raised past the episode length
        # the count is monotonically NON-DECREASING, which is the property the mode is
        # supposed to have. (VRAM is not the constraint here: Stratas holds the KV.)
        history_messages=200,
    )
    advertised = sorted(s["function"]["name"] for s in harness.tool_schemas)
    result = harness.run_episode(init_state_idx=0, episode_id=994)
    transcript = json.loads(Path(result.transcript_path).read_text())["transcript"]
    client.close()
    time.sleep(0.3)

    texts = [e["result"] for e in transcript if e.get("result")]
    joined = "\n".join(policy.sent)
    checks = {
        "工具面是 move/gripper/look/verify": advertised == ["gripper", "look", "move", "verify"],
        "没有像素工具、也没有 approach": all(
            n not in advertised for n in ("aim", "target", "approach", "grasp", "release")
        ),
        "任务消息不带任何画面（开局是盲的）": policy.initial_message_images == 0,
        "没 look 之前不会出现画面": policy.frames_before_look == 0,
        "look 之后才有画面（每张 look 都留下）": policy.max_images_seen >= 2 * policy.looks > 0,
        "look 被拒绝前的试探给出协议重试": sum(1 for e in transcript if e.get("protocol_retry")) >= 1,
        "aim 在本模式被拒（拒绝理由回到模型）": "mode_blocked_tool" in joined,
        "画面不带网格": policy.grid_frames == 0 and policy.frames > 0,
        "三工具仍能完成任务": bool(result.success),
    }
    # THE ONLY THING ALLOWED TO DROP FRAMES IS THE COMPRESSION. With `summary_every=32`
    # (the default) the summariser replaces the whole history at turn 33 -- that IS what
    # compression is, and it reclaims the images with it. So the property is not "monotone"
    # but: within a block the count never decreases, and every decrease lands exactly on 0
    # (a decrease to anything else would be the piecemeal elision of the legacy
    # `image_keep=2` budget, which is what this mode exists to remove).
    series = policy.image_series or [0]
    drops = [i for i, (a, b) in enumerate(zip(series, series[1:])) if b < a]
    # Split AT each drop: a segment is [start .. drop], so the drop itself opens the next
    # one. (First version split on `drops[1:]`, which left the drop inside the only segment
    # and made the "monotone within a block" check fail on a correct series.)
    segments: list[list[int]] = []
    start = 0
    for d in drops:
        segments.append(series[start:d + 1])
        start = d + 1
    segments.append(series[start:])
    kept_within_block = all(all(b >= a for a, b in zip(seg, seg[1:])) for seg in segments)
    drops_are_compressions = all(series[i + 1] == 0 for i in drops)
    # The frames really do accumulate inside a block (peak = 2 per look: agent + wrist), and
    # the drop is the compression reclaiming them -- after which a scripted policy that stops
    # looking reads 0 for the rest of the episode, which is correct, not a leak.
    peak_before_drop = max(series[: (drops[0] + 1) if drops else len(series)])
    checks["历史图片只在压缩时被回收（块内只增不减、掉就掉到 0）"] = (
        kept_within_block
        and drops_are_compressions
        and peak_before_drop >= 2 * policy.calls.count("look")
        and bool(drops)
    )
    print(f"  [诊断] 每轮提示里的图片数序列={series}（掉帧位置={drops}，掉前峰值={peak_before_drop}）")
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"  look 腿：success={result.success} source={result.success_source} "
          f"turns={result.turns} retries={result.protocol_retries} error={result.error!r}")
    print(f"  look 次数={policy.calls.count('look')} 最多同时可见图片={policy.max_images_seen} "
          f"最后一轮提示里图片数={policy.last_seen_images} 收到的帧={policy.frames}")
    print(f"  每轮提示里的图片数序列（前 6 / 后 6）：{policy.image_series[:6]} … {policy.image_series[-6:]}")
    if texts:
        print(f"  示例：{texts[0][:80]}")
    return all(checks.values())


class PlannerStub:
    """One object, two roles (the harness hands the SAME model to the sub-harness).

    It routes on the system prompt, which is the only thing that differs between the
    planner's context and the sub-model's: `规划者` -> the planner script (one dispatch,
    then verify); otherwise -> hand everything to a `RawPolicy`, i.e. the exact scripted
    `look`-mode executor the raw leg already proved. That is precisely the 档 E shape:
    same weights, different context.
    """

    def __init__(self, server: SimServer) -> None:
        self.inner = RawPolicy(server=server)
        self.planner_calls = 0

    def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
        system = str(next((m.get("content", "") for m in messages if m.get("role") == "system"), ""))
        if "规划者" in system:
            self.planner_calls += 1
            task = (
                "抓起 alphabet soup：先张开夹爪，把爪心移到蓝罐正上方，下降到罐身再合爪；"
                "然后抬到 z≥0.25，移到篮子上方，降到 0.16 放手。"
                "**上次失败原因（写给你避免）：合手高度太高会只碰到盖沿——eef z 要 <0.08 再 close。**"
            )
            if self.planner_calls == 1:
                return (
                    "现状：第一回合，还没看过现场。\n"
                    "判定：整体分两个阶段，先交执行者抓取+搬运。\n"
                    "动作：调用 dispatch。\n"
                    + _call("dispatch", {"task": task, "budget_turns": "40"})
                )
            return (
                "现状：执行者已返回报告。\n"
                "判定：验收，请环境判定任务是否完成。\n"
                "动作：调用 verify。\n"
                + _call("verify", {"claim": "task_complete"})
            )
        return self.inner.chat(messages, tools=tools, **kwargs)


def planner_leg(ipc_dir: Path) -> bool:
    """档 E: the planner has NO actuation; the executor is the `look`-mode model.

    What has to be proven here (each is the reason the mode exists):
    1. The planner's surface is exactly {look, dispatch, verify} -- no move, no gripper.
    2. ONE dispatch runs a sub-episode with the look-mode surface and a turn budget,
       WITHOUT resetting the world (the arm carries over; the sub inherits 爪上 state).
    3. The report the planner reads carries turns used / gripper verdicts / final eef.
    4. The sub's transcript is flattened INTO the planner's episode file (depth-marked).
    5. The whole task still completes (the sub IS the proven raw-leg controller).
    """
    server = SimServer(
        ipc_dir=ipc_dir,
        suite_name="libero_object",
        task_id=0,
        cam_size=256,
        observation_mode="planner",
        pixel_grid_px=32,
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 120}, daemon=True)
    thread.start()
    time.sleep(0.2)
    client = SimClient(ipc_dir)
    stub = PlannerStub(server=server)
    harness = Harness(
        client=client,
        model=stub,
        workspace=REPO,
        max_turns=8,
        tool_protocol="v2",
        log=False,
        observation_mode="planner",
    )
    advertised = sorted(s["function"]["name"] for s in harness.tool_schemas)
    result = harness.run_episode(init_state_idx=0, episode_id=993)
    transcript = json.loads(Path(result.transcript_path).read_text())["transcript"]
    client.close()
    time.sleep(0.3)

    sub_entries = [e for e in transcript if e.get("sub")]
    planner_entries = [e for e in transcript if not e.get("sub") and e.get("result")]
    dispatch_reports = [e for e in planner_entries if str(e.get("result", "")).startswith("[dispatch")]
    moves = [e for e in sub_entries if (e.get("structured_result") or {}).get("tool") == "move"]
    grips = [e for e in sub_entries if (e.get("structured_result") or {}).get("tool") == "gripper"]
    st = (dispatch_reports[0].get("structured_result") or {}).get("measurement") or {} if dispatch_reports else {}
    report = str(dispatch_reports[0].get("result", "")) if dispatch_reports else ""
    checks = {
        "planner 工具面 = look/dispatch/verify（无 move/gripper）": advertised == ["dispatch", "look", "verify"],
        "发生了至少一次 dispatch，报告到达 planner": bool(dispatch_reports),
        "报告带预算用量与夹爪回报": (
            "gripper" in report and bool(st.get("sub_turns_budget")) and st.get("sub_turns", 0) >= 1
        ),
        "子回合确实在跑（move+gripper 都在子层）": bool(moves) and bool(grips),
        "子回合已折叠进同一份转写（带 sub 深度标记）": bool(sub_entries)
        and all(e.get("outer_turn") for e in sub_entries),
        # Regression guard for the 162-void (2026-10-12): the planner's task text
        # was logged/reported/audited but never delivered -- the executor opened on
        # the standing sim task every time. The anchor string is unique to the task
        # the stub planner writes above, so its presence in the sub's OPENING message
        # is end-to-end proof the channel now exists.
        "阶段任务确实送达执行者（开场含 dispatch task 锚点）": any(
            "eef z 要 <0.08" in str(e.get("opening_user", "")) for e in sub_entries
        ),
        "执行者没有重置世界（子层首步爪心 = reset 位置之后）": (
            moves and abs((moves[0].get("structured_result") or {}).get("measurement", {}).get("eef_start", [9])[0] - (-0.152)) < 0.03
        ),
        "预算生效（子回合数 ≤ 预算）": (not st) or st.get("sub_turns", 0) <= st.get("sub_turns_budget", 0),
        # Regression guard for the two report bugs the first planner run exposed
        # (2026-10-12): `look` is recorded canonically as `observe`, so the report
        # read "look 0 次" on all 162 dispatches; and the held-state was pasted with
        # its label twice ("爪上：爪上：空").
        "报告里的 look 计数不是恒 0（observe 要按 look 报）": (
            st.get("sub_tools") == _count_sub_tools(sub_entries)
            and _count_sub_tools([{"structured_result": {"tool": "observe"}}]) == {"look": 1}
        ),
        "爪上状态只出现一次标签": "爪上：爪上" not in report,
        # The executor never returns early, so an under-sized budget is a lost phase:
        # the floor is enforced by validation (see contracts.DISPATCH_BUDGET_RANGE).
        "预算下限拦住「做不完的阶段」": C.DISPATCH_BUDGET_RANGE[0] >= 20,
        "planner 自身回合数很小（外层只发号施令）": result.turns <= 4,
        "整体任务完成（执行者=已证明的 look 腿控制器）": bool(result.success),
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"  planner 腿：success={result.success} source={result.success_source} "
          f"外层 turns={result.turns}；dispatch 数={len(dispatch_reports)}；"
          f"子条目 {len(sub_entries)}（move {len(moves)} / gripper {len(grips)}）")
    if dispatch_reports:
        print(f"  报告节选：{report[:180]}")
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
    print("\n=== 档 B 像素腿（pixels 模式 aim/target，无 LLM）===")
    ok = pixel_leg(REPO / "runtime" / "ipc_dryrun3") and ok
    print("\n=== raw 腿（唯 move/gripper/verify，无定位工具，无网格，无 LLM）===")
    ok = raw_leg(REPO / "runtime" / "ipc_dryrun4") and ok
    print("\n=== 阶梯腿（档 D：move 只有 far/near/fine 三档）===")
    ok = ladder_leg(REPO / "runtime" / "ipc_dryrun5") and ok
    print("\n=== look 腿（model 自己决定何时看图，历史图不清理）===")
    ok = look_leg(REPO / "runtime" / "ipc_dryrun6") and ok
    print("\n=== planner 腿（档 E：主模型只规划+验收，执行者=look 模式子模型）===")
    ok = planner_leg(REPO / "runtime" / "ipc_dryrun7") and ok
    print("PLUMBING OK" if ok else "PLUMBING PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
