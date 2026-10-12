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
from arm_agent.agent.harness import Harness
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


def raw_next_call(server: SimServer) -> tuple[str, dict[str, Any]]:
    """The next `move`/`gripper` call a controller would make, from TRUTH.

    Privileged on purpose (same as the other legs): this answers "are two tools enough
    to do the task at all", which is a plumbing claim. Whether a MODEL can do it with
    only those two tools is the experiment this leg makes possible, not the thing it
    measures.

    The shape of the controller is also the recipe the `raw` prompt teaches: one small
    step, then look again (the state is re-read from the simulator on every turn).
    """
    obs = server.sim._obs
    eef = np.asarray(obs[C.OBS_EEF_POS], dtype=float)
    objects = server.sim.pack(obs)["objects"]
    can = np.asarray(objects["alphabet_soup_1"], dtype=float)
    basket = np.asarray(objects["basket_1"], dtype=float)
    width = float(np.abs(np.asarray(obs[C.OBS_GRIPPER_QPOS])).mean())
    carried, _ = server.adapter.held_object()

    def toward(target: np.ndarray, tol: float) -> tuple[str, dict[str, Any]] | None:
        dx, dy = target[0] - eef[0], target[1] - eef[1]
        if abs(dx) > tol or abs(dy) > tol:
            axis, delta = ("x", dx) if abs(dx) >= abs(dy) else ("y", dy)
            step = min(5.0, max(1.0, abs(delta) * 100.0))
            return "move", {"direction": f"{'+' if delta > 0 else '-'}{axis}",
                            "distance_cm": f"{step:.1f}"}
        return None

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
                return "move", {"direction": "+z", "distance_cm": "5"}
            return step
        if eef[2] > can[2] + 0.030:
            return "move", {"direction": "-z", "distance_cm": "3"}
        return "gripper", {"action": "close"}
    # carrying: up, across, down, open, and let the ENV judge.
    step = toward(basket, tol=0.015)
    if step:
        if eef[2] < 0.25:
            return "move", {"direction": "+z", "distance_cm": "10"}
        return step
    if eef[2] > 0.20:
        return "move", {"direction": "-z", "distance_cm": "5"}
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

    def __init__(self, server: SimServer) -> None:
        self.server = server
        self.sent: list[str] = []
        self.calls: list[str] = []
        self.grid_frames = 0
        self.frames = 0
        self.opens = 0
        self.done = False

    def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
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
        done_calls = sum(1 for c in self.calls if c not in ("aim", "target", "approach"))
        if len(self.calls) < len(self.PRELUDE):
            name, args = self.PRELUDE[len(self.calls)]
            self.calls.append(name)
            return "现状：脚本化步骤。\n判定：试探被禁工具。\n动作：调用工具。\n" + _call(name, args)
        if done_calls == 0:
            # The prelude was refused, so no tool has actually run yet; nothing to do
            # but keep asking for a legal one.
            pass
        if self.done:
            name, args = "verify", {"claim": "placed"}
        else:
            name, args = raw_next_call(self.server)
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
    print("PLUMBING OK" if ok else "PLUMBING PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
