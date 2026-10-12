"""The agent loop: observe -> think -> act -> feed back -> verify.

M1 scope: a single episode, 8 tools, no summarization compression yet. What IS
here from day one (each item is a measured lesson, see DESIGN.md):

  * exactly one tool call per turn; 0 or >1 gets re-prompted (E5);
  * replies are truncated to the first </tool_call> (E5 eos overrun);
  * every action result carries MEASURED numbers (displacement, position,
    gripper width) — the model's free-form direction choices were wrong twice
    in E5, so feedback + retry is the correction channel, not hoping;
  * old images elided down to a small recency window (processor re-encodes
    every frame each turn otherwise);
  * a full transcript is written per episode for replay/analysis.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from arm_agent import contracts as C
from arm_agent.agent.llm import prepare_for_model
from arm_agent.agent.parser import (
    ToolCall, as_float, parse_tool_calls, truncate_to_first_call, validate_calls,
)
from arm_agent.agent.prompts import (
    get_system_prompt,
    get_tool_schemas,
    ladder_appendix,
    mode_appendix,
)
from arm_agent.agent.tool_protocol import canonical_name, structured_result
from arm_agent.agent.render import ACCENT, EpisodeRecorder, GREEN, RED, YELLOW

# Key of the server's magnified crop, shared through `contracts` so neither process
# has to import the other (the agent env has no robosuite). If this ever drifts from
# the server's key, the zoom silently stops being attached and the model is left
# judging a ring against an object at 1x while its prompt promises a magnifier.
ZOOM_KEY = C.ZOOM_KEY

SKILLS_DIRNAME = "skills"
# Consecutive identical tool calls (same name, byte-identical args) after which the
# episode is ENDED and LABELLED instead of burning the remaining turns.
#
# MEASURED (aim v3, episode 0): 76 identical `aim(58,134)` calls -- 95% of the episode,
# with the model's 现状 text frozen word-for-word. The 3-call hint fired 37 times and was
# ignored, so a note alone does not break this loop. MAXTURNS is the blunt version of
# the same guard (it ends the episode after 80 turns and says only "no success"); this
# one ends it earlier AND names the reason, which is the difference between a silent
# timeout and a diagnosable failure.
REPEAT_STALL_LIMIT = 12
# A repeated call is only a STALL if nothing moved. The threshold is set from the
# MEASURED behaviour of the two ends of the scale, not picked: a blocked `move` reports
# an exact (0,0,0) displacement, while the smallest rung (档 D `fine`, a 1mm request)
# advances 0.6-0.8mm per call -- the servo stops inside its own 0.5mm arrival window,
# so it under-delivers every time and grinds forward slowly. 12 of those is 7mm of real
# progress, not a loop, and killing the episode there was wrong.
STALL_MOTION_M = 0.0003
# Kept equal to `action_adapter.DESCEND_Z_TOL_M`: a descent that stops further
# than this from the commanded z did not arrive, and saying it did is what makes
# the model close on a rim instead of re-descending.
DESCEND_SHORTFALL_M = 0.005
# The shared skill file, injected into the executor's system prompt.
SHARED_SKILL_FILE = "scene_tools.md"
# Feed-back images: the newest N frames stay as pixels, the rest become text
# stubs. Was 1 while only the agent view was attached; the loop now attaches the
# agent AND wrist cameras each step, so 2 keeps exactly one whole frame pair
# (keeping 1 would silently throw away half of the newest pair). Tokens: 2 x
# 66 = 133 per turn, and only for the newest turn.
HISTORY_IMAGE_KEEP = 2
# Conversation cap. MEASURED (probe_gpu_memory.py): the OOM allocation was
# exactly one full-attention N x N matrix (2900^2 * 16 heads * 2 bytes = 266 MiB)
# — on this Turing card SDPA has no mem-efficient kernel, so attention memory
# grows quadratically with the token count. A 24-message cap did NOT prevent the
# turn-12 OOM (a 12-turn tail is still ~3000 tokens).
#
# v7 RE-MEASURED the ceiling: at 10 messages the run survived to turn 62 and then
# died asking for 274 MiB. Tool results are long (a `locate` reply is ~350 CJK
# chars), so "10 messages" is not 10 short messages -- the tail keeps growing with
# the system prompt. 6 messages (system + task + 2 turn pairs) buys back ~1000
# tokens, i.e. ~250 MiB of N x N matrix, and costs little: every `locate`
# restates the full geometry, so the model rarely needs a longer tail.
#
# v15 RE-MEASURED again, from the other direction: adding the slip/"dropped"
# warnings made results ~120 chars longer each, and at 6 messages that OOM'd at
# turn 19 (260 MiB request, 261 MiB free). Lesson repeated: **changing the prompt
# IS changing the VRAM budget** -- run probe_token_budget2.py before a GPU run.
# 5 messages keeps the tail at one turn pair plus the note. v16 then went to 4:
# probe_token_budget2 measured 5 messages + the long warnings at 4219 tokens
# (543 MiB matrix) versus the 4446/603 MiB that OOM'd, i.e. only 60 MiB of slack
# -- not enough. 4 messages = 3992 tokens = 486 MiB.
#
# 2026-10-10, RE-MEASURED ON THE 32 GiB CARD (RTX 5090, single node, v2 tools):
# the cap was still 4, and `ctx` sat FLAT at 1263-1285 tokens for all 80 turns
# while the peak was 7.62 GiB of 32 GiB. The run never came near the ceiling --
# and still could not see its own history: the same 4-beat loop repeated for 80
# turns because every cross-turn cause had been trimmed away. A cap that is
# never reached is not a safety margin, it is a blindfold. Raised to the tier
# docs/context_budget.md recommends for 32 GiB (MSG=32). The 10 GiB tier is
# still one flag away (`--history-messages 4`). Note that on sm_120
# (RTX 5090) flash-attention IS available, so the B/tok^2 term is ~4x smaller
# than the 170.7 measured on sm_75; N^2 is not the binding constraint here.
MAX_HISTORY_MESSAGES = 32


@dataclass
class EpisodeResult:
    task: str = ""
    language: str = ""
    success: bool = False
    # "env" = the sim's own check_success() flipped True (the official LIBERO
    # criterion); "declared" = the agent called verify and the sim agreed at
    # that instant. Both count as success; the field exists so a report can
    # never confuse "the object is placed" with "the agent noticed".
    success_source: str = ""
    done_declared: bool = False
    turns: int = 0
    tool_calls: int = 0
    protocol_retries: int = 0
    elapsed_s: float = 0.0
    transcript_path: str = ""
    error: str = ""


class Harness:
    def __init__(
        self,
        client: Any,
        model: Any,
        workspace: Path,
        max_turns: int = 60,
        attach_images: bool = True,
        log: bool = True,
        prompt_mode: str = "full",
        tool_protocol: str = "v1",
        history_messages: int | None = None,
        image_keep: int | None = None,
        observation_mode: str = "truth",
        move_ladder: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.workspace = Path(workspace)
        self.max_turns = max_turns
        self.attach_images = attach_images
        self.log = log
        # Both budgets are overridable per run: the right cap depends on the
        # card (see MAX_HISTORY_MESSAGES). None = use the module default.
        self.history_messages = int(history_messages) if history_messages else MAX_HISTORY_MESSAGES
        # `look` mode sets auto_frames/image_keep, so the mode has to be resolved FIRST.
        # (Cost one debugging round: the image budget was computed before the mode existed
        # and the ctor raised AttributeError.)
        # 档 B: `truth` = the simulator owns object positions (the A/B baseline);
        # `pixels` = it does not, and the model targets pixels instead.
        self.observation_mode = str(observation_mode).strip().lower()
        self.auto_frames = self.observation_mode != "look"
        self.image_keep = int(image_keep) if image_keep is not None else (
            # -1 = NEVER elide a frame from the history (`llm.prepare_for_model`). In
            # `look` mode that is the whole point: the model paid a turn for each frame,
            # and comparing the frame it took before a step with the one after is how it
            # judges whether the step was too big.
            -1 if self.observation_mode == "look" else HISTORY_IMAGE_KEEP
        )
        # "full" = the accumulated scene-specific prompt; "plain" = task +
        # frame + tool semantics only (see prompts.get_system_prompt).
        self.prompt_mode = "v2" if tool_protocol == "v2" and prompt_mode == "full" else prompt_mode
        self.tool_protocol = tool_protocol
        # 档 D: with the ladder on, `move` takes a NAMED rung (far/near/fine = 10cm/1cm/1mm)
        # instead of a distance, so the magnitude is code-owned. The value the server
        # receives is still `distance_cm` -- the ladder lives entirely on the agent
        # side, because the simulator was never the thing that could not do the
        # arithmetic. See contracts.MOVE_LADDER_CM.
        self.move_ladder = bool(move_ladder)
        # PIXEL-GAP scale, in mm per pixel at the agent view. Starts at the MEASURED
        # 2.96-4.04 mm/px and SELF-CALIBRATES: every motion result carries the simulator's
        # own `mm_per_px` at the depth it just reached, so after one step this is exact
        # instead of nominal. This is the number the model cannot compute (its px->cm
        # arithmetic slips a factor of ten, measured) and the code can.
        self.mm_per_px = C.GLOBAL_MM_PER_PX_DEFAULT
        self.tool_schemas = get_tool_schemas(
            tool_protocol, self.observation_mode, move_ladder=self.move_ladder
        )
        self.legacy_tools = str(tool_protocol).lower() != "v2"
        self.skills_dir = self.workspace / SKILLS_DIRNAME
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        # SHARED SKILL, injected into the executor's system prompt.
        #
        # It lives on disk (not inline in prompts.py) so it can be edited /
        # self-iterated without touching code. Read once here: the system prompt
        # string is rebuilt every turn, and re-reading a file each turn would be
        # pure I/O overhead.
        #
        # History: it used to be shared with the release-gate verifier because the
        # executor had the axis convention in its own prompt while the verifier
        # did not, and the verifier's picture-relative hints ("篮子在右侧") made
        # the executor march in +x while the real deficit was +y -- 40/40 moves in
        # the wrong axis, measured. The gate is gone (2026-10-11), the axis
        # convention stays: it is scene vocabulary, and it is the ONE thing here
        # that is a description of the world rather than a decision taken for the
        # model.
        skill_path = self.skills_dir / SHARED_SKILL_FILE
        self.shared_skill = f"\n\n{skill_path.read_text().strip()}" if skill_path.exists() else ""
        # Grasp state, taken from the gripper's own verdict ("holding" in the
        # adapter note) and NEVER inferred from distance. MEASURED (P0a): the
        # proximity rule reported the pads parked 5.5 cm above an untouched can
        # as holding it -- 7 "slip" + 5 "dropped" false alarms in one run.
        #
        # This is the loop's ONLY piece of cross-turn state, and it stays because
        # the observation carries no contact sensor: `format_state` needs it to
        # decide whether to look for an object in the pads at all.
        #
        # DELETED 2026-10-10: `_prev_gap` (grip-slip trend / "物体掉了" event),
        # `_prev_grasped` (the two self-harm warnings) and `_last_width` (the
        # "you closed on already-shut pads" fact). Each derived an event the
        # model can now read straight out of its own history -- and each was
        # justified in its comment by "the model cannot see this by itself,
        # because history is capped at MAX_HISTORY_MESSAGES". That cap is 32
        # now, so the premise is gone; what is left is cross-turn bookkeeping
        # that has already produced one false alarm that cost a whole episode
        # (a deliberate `open` reported as "物体掉了", 2026-10-10).
        self._grasped = False
        # The last pixel the model asked to `aim` at, with the camera it asked on.
        # Used for one thing only: telling it that an identical aim cannot produce
        # new evidence. Cleared by every tool that can move the arm or the scene.
        self._last_aim: tuple[str, int, int] | None = None
        # Cross-turn duplicate detector (an identical tool call with identical args).
        # See the note where it is consumed: its only job is to state a fact about
        # the model's own repeated call, never to block or replace it.
        self._repeat_key: tuple[str, str] | None = None
        self._repeat_n = 0
        # Which container the executor last ran `approach` on. Feeds the
        # arithmetic instrument below (log only, never blocks).
        self._last_approach_target = ""
        self._approach_order: list[str] = []
        self._approached: dict[str, list[float]] = {}
        # What is between the pads, as a SIM BODY NAME, taken from the contact
        # table (adapter.held_object). None = nothing held. This replaces the
        # ground-truth-distance guess `_carried_name` as the primary source; the
        # distance version survives only as a fallback for the legacy surface.
        self._held_body: str | None = None
        # DELETED 2026-10-11: the two-role release gate (`--verifier`, a
        # fresh-context verifier that had to approve every carrying `open`).
        #
        # Why it existed: measured 2026-10-10, the executor asserted arrival
        # without checking (5/5 releases proceeded on a false "已到达篮子正上方")
        # and a second role asked the same question answered correctly.
        #
        # Why it is gone: the failure mode it guarded (a release that leaves the
        # payload ON the basket rim) was a WRONG CONSTANT, not a missing check --
        # the skill had the rim at 0.115 while the basket's own geometry says
        # 0.1513, and every `approach(basket, 20)` therefore dropped the can on
        # the rim. With the constants measured, the gate's own record is 16/16
        # releases approved, 0 blocks, and an independent audit of those 16
        # releases (scripts/audit_release_poses.py) found 16/16 CORRECT poses
        # (|Δx|,|Δy| <= 0.2 cm, payload bottom 0.195-0.198 vs rim 0.151). So it
        # cost one prefill of the context plus a reply per release (~4 % of wall
        # clock) to re-check a solved problem -- and on the 9B it had a
        # documented 4/4 false-REJECT record against a provably correct pose.
        #
        # The DIRECTION of this project is to hand judgement back to the model
        # and keep the harness as measurement only, so the rejection path goes;
        # `_gate_arithmetic` stays as a logged instrument (it is free: it reads
        # numbers the observation already carries).

    # ---------------------------------------------------- release instrument
    def _gate_container(
        self, obs: dict[str, Any] | None
    ) -> tuple[str | None, tuple[float, float] | None]:
        """The container the executor means to release INTO, and its centre.

        WHY this is not simply `_last_approach_target`: MEASURED (run 16,
        2026-10-10). The executor's most recent `approach` is usually the PAYLOAD
        (it re-approaches the can after every failed drop). The payload rides in
        the gripper, so comparing the gripper against it gives Δ ≈ 0 every time
        and the gate approved 10/10 releases -- including ones 18 cm short of the
        basket. The destination container is the most recent `approach` target
        that is NOT the object currently between the pads.
        """
        if not obs:
            return None, None
        held = self._held_body or (_carried_name(obs) if self._grasped else None)
        objects = obs.get("objects") or {}
        for name in reversed(self._approach_order):
            if name == held:
                continue
            pos = objects.get(name)
            if pos and len(pos) >= 2:
                return name, (float(pos[0]), float(pos[1]))
            rec = self._approached.get(name)
            if rec:
                return name, (rec[0], rec[1])
        return None, None

    def _gate_arithmetic(self, obs: dict[str, Any] | None) -> str:
        """The harness's OWN reading of the release pose -- LOGGED, never
        enforced.

        WHY it survives the verifier's deletion: it is the only independent
        record of where a release happened, and it costs nothing (it subtracts
        numbers the observation already carries). Reading it back is how the
        "the can went in but the run scored a failure" incident was found, and
        `scripts/audit_release_poses.py` re-derives the same numbers from the
        transcripts afterwards. Enforcing it would be the harness deciding for
        the model, which is the opposite of where this project is going: it is
        an instrument now, not a gate.

        Thresholds are the sim's own geometry, measured 2026-10-11 by
        scripts/probe_rim.py + scripts/probe_release_offset.py: the basket's rim
        is at z=0.1513 (body z=-0.0046 + a 17.3 cm box), the can is 8.5 cm tall,
        and the gripper holds it at mid-height so the payload's bottom rides
        0.050 below the EEF.  The old pair (rim 0.115, bottom = eef - 0.054) was
        3.6 cm out, and every release at approach(basket, 20) then left the can
        resting ON the rim.
        """
        if not obs:
            return "没有 obs"
        eef = obs.get("eef_pos")
        name, centre = self._gate_container(obs)
        if eef is None or centre is None:
            held = self._held_body or (_carried_name(obs) if self._grasped else None)
            return f"无法判断（容器未知，最近 approach 的是{held or '无' }）"
        dx = (eef[0] - centre[0]) * 100.0
        dy = (eef[1] - centre[1]) * 100.0
        bottom = eef[2] - 0.050
        ok = abs(dx) <= 5.0 and abs(dy) <= 5.0 and bottom >= 0.151
        return (
            f"容器={name} Δx={dx:+.1f}cm Δy={dy:+.1f}cm 爪心z={eef[2]:.4f} "
            f"物体底≈{bottom:.4f} ⇒ {'应放行' if ok else '不应放行'}"
        )

    # -------------------------------------------------------------- episode
    def run_episode(
        self,
        init_state_idx: int = 0,
        episode_id: int = 0,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        result = EpisodeResult()
        t_start = time.time()
        transcript: list[dict[str, Any]] = []
        last_images: dict[str, Any] = {}
        try:
            resp = self.client.reset(init_state_idx=init_state_idx)
            obs = resp.obs
            images = resp.images(self.client.ipc_dir)
            last_images = images
            result.task = obs.get("task", "")
            result.language = obs.get("language", "")
            if recorder is not None:
                recorder.task = result.language or result.task
                recorder.record(
                    0,
                    tool_call="(episode start)",
                    result=f"任务: {result.language}\n初始状态: {format_state(obs)}",
                    status="开始",
                    status_color=ACCENT,
                    images=last_images,
                )

            # `look` mode: the task message carries NO frame. Seeing is an action the model
            # takes, and the very first thing it must do is decide to take it -- otherwise
            # it opens with two frames of context it did not ask for and never learns that
            # looking is something it controls.
            initial_images = [
                *(
                    [{"type": "image", "image": images[cam]} for cam in ("agent", "wrist") if images.get(cam) is not None]
                    or [{"type": "image", "image": images["agent"]}]
                )
            ] if self.auto_frames else []
            initial_note = (
                "[上图：全局相机 + 腕部相机（从上往下看夹爪下方，用来判断自己在哪）]\n"
                if self.auto_frames
                else "[本模式不附画面：想看图就调用 look。]\n"
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": get_system_prompt(self.prompt_mode, self.observation_mode, self.move_ladder) + self.shared_skill + mode_appendix(self.observation_mode) + ladder_appendix(self.move_ladder) + self._skills_index()},
                {
                    "role": "user",
                    "content": [
                        *initial_images,
                        {
                            "type": "text",
                            "text": (
                                f"任务：{result.language}\n"
                                f"{initial_note}"
                                f"{format_state(obs)}\n"
                                "请开始。记住：每次只调用一个工具。"
                            ),
                        },
                    ],
                },
            ]

            for turn in range(1, self.max_turns + 1):
                result.turns = turn
                t_turn0 = time.time()
                # Two budgets, both from measurements (see MAX_HISTORY_MESSAGES and
                # HISTORY_IMAGE_KEEP): cap the transcript so attention stays cheap
                # (quadratic on this card), then elide all but the newest frames so
                # the processor stops re-encoding the whole visual history.
                t_llm0 = time.time()
                try:
                    reply = self.model.chat(
                        prepare_for_model(
                            compact_history(messages, self.history_messages),
                            self.image_keep,
                        ),
                        tools=self.tool_schemas,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Measured failure modes that must not lose the episode: CUDA
                    # OOM in the vision tower / attention. Record and stop cleanly
                    # so the transcript survives for diagnosis.
                    result.error = f"model.chat failed at turn {turn}: {type(exc).__name__}: {exc}"
                    self._say(f"[turn {turn}] FATAL: {result.error}")
                    break
                reply = truncate_to_first_call(reply)
                llm_s = time.time() - t_llm0
                calls = parse_tool_calls(reply)
                messages.append({"role": "assistant", "content": reply})
                transcript.append({"turn": turn, "assistant": reply})

                call_errors = validate_calls(
                    calls,
                    legacy=self.legacy_tools,
                    observation_mode=self.observation_mode,
                    move_ladder=self.move_ladder,
                )
                if call_errors:
                    result.protocol_retries += 1
                    # Tell a CUT-OFF reply apart from one that simply has no tool
                    # call. MEASURED (2026-10-10, 80-turn run): 18 of 19 retries
                    # were truncations at `--max-new-tokens 160` -- the model's
                    # Chinese preamble consumed the budget and the XML never
                    # reached `</tool_call>`. `got_0` alone hid that.
                    if not calls and "</tool_call>" not in reply:
                        call_errors = [
                            "reply_cut_off_before_tool_call_closed:"
                            "raise --max-new-tokens or write less preamble"
                        ]
                    note = (
                        f"工具协议错误：{', '.join(call_errors)}。"
                        "请只输出一个合法工具调用，并修正参数。"
                    )
                    if any(str(e).startswith("mode_blocked_tool:") for e in call_errors):
                        note += (
                            " 本观测模式里没有物体坐标可用：请改用 aim 确认像素、再用 target 移动"
                            "（在带网格的面上读，例：u=128, v=96），或用 move 沿世界轴小步移动。"
                        )
                    if any(str(e).startswith("invalid_enum:camera") for e in call_errors):
                        # The wrist camera is mounted on the hand: a fixed pixel there
                        # is a fixed point in the hand's frame, so it can never be a
                        # target (measured: the grounded point travels with the
                        # gripper). Saying WHY saves a turn of blind retries.
                        note += (
                            " 腕部相机装在夹爪上，不能用来指定目标（固定像素=相对手恒定的点，会跟着夹爪一起动）；"
                            "给像素请用 camera=agent，腕部画面只用来判断方向，再用 move 沿世界轴走 2~5cm。"
                        )
                    messages.append({"role": "user", "content": note})
                    transcript.append({"turn": turn, "protocol_retry": True})
                    self._say(f"[turn {turn}] protocol retry (calls={len(calls)})")
                    if recorder is not None:
                        recorder.record(
                            turn,
                            reply=reply,
                            tool_call=f"(协议失败: {', '.join(call_errors)})",
                            result=note,
                            status="协议重试",
                            status_color=YELLOW,
                            images=last_images,
                        )
                    continue

                call = calls[0]
                result.tool_calls += 1
                self._say(f"[turn {turn}] {call.name} {call.args}")

                # DELETED 2026-10-11: the release-gate interception (see the
                # constructor note). What replaces it is one log line: when the
                # executor is about to release a payload, the harness writes down
                # where the gripper actually is. That is measurement, not a
                # decision -- the action is ALWAYS dispatched.
                if (
                    canonical_name(str(call.name)) == "gripper"
                    and str((call.args or {}).get("action", "")).strip().lower() == "open"
                    and self._grasped
                ):
                    self._say(f"[turn {turn}] 放手位置自检: {self._gate_arithmetic(obs)}")

                try:
                    t_tool0 = time.time()
                    outcome = self._dispatch(call, obs)
                    outcome["result"] = self._structured_outcome(call, outcome)
                    tool_s = time.time() - t_tool0
                    # A DEGENERATE LOOP IS A FACT ABOUT THE MODEL'S OWN CALLS, and
                    # saying so is not a decision. MEASURED (aim v2, episode 0): after a
                    # failed `verify` the model repeated the IDENTICAL call 26 times in
                    # a row and spent a third of the episode on it -- its 现状/判定 text
                    # froze word-for-word at turn 56 and never moved again, because a
                    # failed verdict carries no information the model can act on. The
                    # harness cannot tell it WHY the task is not done (that would be
                    # scene truth), but it can tell it that this call has already been
                    # made N times with nothing changing in between.
                    rep_key = (
                        canonical_name(str(call.name)),
                        json.dumps(sorted((str(k), str(v)) for k, v in (call.args or {}).items()),
                                   ensure_ascii=False),
                    )
                    if rep_key == self._repeat_key:
                        self._repeat_n += 1
                    else:
                        self._repeat_key, self._repeat_n = rep_key, 1
                    # ...UNLESS THE ARM ACTUALLY MOVED. MEASURED (档 D dry run): a `+z near`
                    # raise moved 8.3-9.3mm per call, twelve identical times in a row, and
                    # the guard killed an episode that was working -- its note ("这期间没有
                    # 任何动作，测量不会变化") would have been FALSE, and a false instrument is
                    # worse than no instrument: the model reads it as "the arm is stuck" and
                    # starts inventing workarounds. So the counter only counts calls that
                    # changed nothing.
                    if moved_m(outcome) >= STALL_MOTION_M:
                        self._repeat_n = 1
                    if self._repeat_n >= 3:
                        # The ARGS are echoed on purpose. MEASURED (aim v3, episode 0):
                        # the model repeated `aim(58,134)` 76 times while its own 现状
                        # line said the ring was at (60,138) -- it was reading its
                        # earlier TEXT instead of the new frame, so "this call is a
                        # repeat" was not a fact it could check. Restating the exact
                        # arguments restates what it actually asked for.
                        args_txt = ", ".join(f"{k}={v}" for k, v in sorted((call.args or {}).items()))
                        outcome["text"] += (
                            f"（提示：这是第 {self._repeat_n} 次完全相同的 {call.name!s} 调用，"
                            f"参数 {args_txt} 与上一轮逐字相同；这期间没有任何动作，测量不会变化。）"
                        )
                    if self._repeat_n >= REPEAT_STALL_LIMIT:
                        result.error = (
                            f"stalled_repeat: 连续 {self._repeat_n} 次完全相同的 "
                            f"{call.name!s} 调用（参数 {sorted((call.args or {}).items())}）"
                        )
                        self._say(
                            f"[episode] 卡死退出：连续 {self._repeat_n} 次相同的 {call.name!s} 调用"
                        )
                        break
                    # Remember which container the executor last tried to align
                    # with: the release instrument needs that container's
                    # MEASURED centre, and the executor's most recent `approach`
                    # is usually the payload itself (re-approaching the can after
                    # every failed drop), against which Δ ≈ 0 always.
                    if canonical_name(str(call.name)) == "approach":
                        _tgt = str((call.args or {}).get("target", "")).strip()
                        if _tgt:
                            self._last_approach_target = _tgt
                            if _tgt in self._approach_order:
                                self._approach_order.remove(_tgt)
                            self._approach_order.append(_tgt)
                            _req = (outcome.get("result") or {}).get("requested") or {}
                            _xy = _req.get("target_xy")
                            if _xy:
                                self._approached[_tgt] = [float(_xy[0]), float(_xy[1])]
                except Exception as exc:  # noqa: BLE001 - tool errors are prompts, not crashes
                    text = f"[工具错误] {type(exc).__name__}: {exc}。"
                    messages.append({"role": "user", "content": text})
                    transcript.append({"turn": turn, "tool_error": text})
                    if recorder is not None:
                        recorder.record(
                            turn,
                            reply=reply,
                            tool_call=_fmt_call(call),
                            result=text,
                            status="工具错误",
                            status_color=RED,
                            images=last_images,
                        )
                    continue
                # Tag every outcome with the TOOL the model actually called, and
                # key the bookkeeping below off THAT.
                #
                # Do NOT key it off `outcome["kind"]`. That string is echoed back
                # by the action adapter and has already diverged from the names
                # used here: the adapter reports `descend_to` (see
                # action_adapter.ActionOutcome kind="descend_to") while the check
                # below tested for `descend`. MEASURED (P0d AND P0e): the
                # "你手里已经夹着它" warning fired **0 times** in both runs even
                # though the model descended onto its own payload at t5 in both.
                # The P0d commit added the missing `kind` key and still changed
                # nothing, because the VALUE was wrong, not the key. The tool name
                # is a string this module owns (it is the same name the model is
                # shown), so it cannot drift with the adapter.
                outcome.setdefault("tool", call.name)
                # ONE vocabulary for all the bookkeeping below: the CANONICAL tool
                # name. The model calls `gripper`/`approach` under v2 while the
                # legacy surface calls the same actions `set_gripper`/`descend_to`.
                # Comparing the raw name made the grasp-state update and both
                # self-harm warnings DEAD CODE under v2: a `gripper close` that
                # reported "holding" still left `_grasped` False, so `format_state`
                # would keep saying 爪上：空 while the object was in the pads --
                # i.e. the model would believe it had dropped what it was holding.
                # Same class of bug as P0d (a value from another vocabulary), found
                # this time by reading, before it could mislead a run.
                _tool = canonical_name(str(outcome.get("tool", "")))
                obs = outcome["obs"]
                # Ground truth for this turn, computed once: see the long note
                # where it is consumed below.
                sim_done = bool(obs.get("success"))
                _turn_success = (bool(outcome.get("done")) and bool(outcome.get("success"))) or sim_done
                if outcome.get("images"):
                    last_images = outcome["images"]
                rec_s = 0.0
                if recorder is not None:
                    t_rec0 = time.time()
                    recorder.record(
                        turn,
                        reply=reply,
                        tool_call=_fmt_call(call),
                        result=outcome["text"],
                        status="任务成功" if _turn_success else "运行中",
                        status_color=GREEN if _turn_success else ACCENT,
                        images=last_images,
                    )
                    rec_s = time.time() - t_rec0
                # Per-turn timing breakdown (measured, not guessed): the bench
                # says a turn should be ~10 s of LLM work, but real episodes
                # ran at 31-38 s. This line localises the missing time.
                self._say(
                    f"[turn {turn}] timing: llm={llm_s:.1f}s tool={tool_s:.1f}s "
                    f"rec={rec_s:.1f}s total={time.time() - t_turn0:.1f}s"
                )

                if outcome["done"]:
                    result.done_declared = True
                    result.success = bool(outcome["success"])
                    result.success_source = "declared" if result.success else ""
                    result_text = outcome["text"] + "\n" + json.dumps(
                        outcome["result"], ensure_ascii=False, separators=(",", ":")
                    )
                    messages.append({"role": "user", "content": result_text})
                    transcript.append({"turn": turn, "result": result_text, "success": result.success})
                    if result.success:
                        break
                    continue

                # The sim's own verdict rides along in EVERY observation
                # (`pack()` includes `success`) and is refreshed on every
                # physics tick, so the ground truth is already here for free.
                #
                # Until 2026-10-11 it was sampled only inside `verify`, i.e. the
                # metric answered "did the agent declare correctly?" instead of
                # LIBERO's own question, "is the object placed?".  MEASURED on
                # the 16-episode A/B: all 3 failures had the can resting in the
                # basket at some turn (can centre z = 0.0554, the exact value a
                # clean-release probe leaves, and inside the `contain_region`
                # box: xy half-extent 6.1 cm, z ceiling 0.137) -- yet they were
                # scored failures, because the agent had already spent its one
                # declare and either never tried again (B/3, B/4) or went on to
                # DRAG the can back out of the basket (A/1: can pushed from
                # y=0.262 to y=0.547).  Counting them is also what stops the
                # harness from letting a finished episode be wrecked.
                # (`sim_done` itself is computed above, next to the recorder.)

                # Track what is between the pads. Since 档 A this comes from the
                # CONTACT table (`outcome["held"]`), not from a substring of the
                # adapter note: MEASURED at reset, the pads sit half closed
                # (width 0.0207, inside the "holding" band) with ZERO finger
                # contacts, so the old width test reported a grasp on an empty
                # hand at the very start of every episode.
                if _tool in ("grasp", "release", "gripper") and outcome.get("held") is not None:
                    # ONE rule for all three surfaces: the CONTACT table answers "is
                    # anything between the pads". The old `gripper` path asked the
                    # finger WIDTH (`"holding" in note`), which reported an empty hand
                    # as holding at every reset (pads start half closed at 0.0207 with
                    # zero contacts) -- MEASURED, and the reason 档 A moved the verdict
                    # to contact in the first place. `raw` mode exposes the raw action,
                    # but it does not re-expose the wrong sensor.
                    self._grasped = bool(outcome["held"])
                    self._held_body = _object_key_from_body(
                        (outcome.get("measurement") or {}).get("object_body")
                    )
                elif _tool == "gripper":
                    _action = (outcome.get("requested") or {}).get("action")
                    _note = str(outcome.get("note", ""))
                    if _action == "open":
                        self._grasped = False
                    elif _action == "close":
                        # "holding" is the adapter's verdict; "empty" is air.
                        self._grasped = "holding" in _note

                # Deleted 2026-10-10: two cross-turn "self-inflicted harm"
                # warnings (you are squeezing your own payload / you are pressing
                # it into the table). Their stated justification was "the model
                # cannot infer this from the numbers" -- true while history was
                # capped at 4 messages, false now that it holds 32: the previous
                # turn's `close` verdict and the model's own last action are both
                # still in the conversation. They were also the most prescriptive
                # text in the loop ("可能把它挤出去"), which is exactly what v2 is
                # supposed to leave to the model. The FACTS they depended on
                # (open before close; do not descend onto your own payload) now
                # live in the system prompt instead.
                content: list[dict[str, Any]] = []
                # In `look` mode a frame is attached ONLY when the model asked for one:
                # `self.auto_frames` is False there, so the guard needs the second clause.
                if self.attach_images and (
                    self.auto_frames or canonical_name(str(call.name)) in ("observe", "look")
                ):
                    # BOTH cameras, every step that has fresh frames.
                    #
                    # MEASURED 2026-10-10: only the agent view was ever attached,
                    # so "am I actually above the basket?" had to be inferred from
                    # a third-person 256 px frame. The model released the can
                    # 23 cm short (run 5, t9) and then 7 cm short from 10 cm of
                    # drop height (run 6, t56), which rolled it out of the arm's
                    # reach for the rest of the episode.
                    #
                    # The wrist camera looks straight down at what is under the
                    # pads -- exactly the missing piece. Cost, measured with the
                    # real processor (patch 16 / merge 2): a 160 px frame is
                    # grid [1,16,16] = 64 image tokens + 2 delimiters = 66, so the
                    # second camera adds ~67 tokens PER TURN, and only for the
                    # newest turn -- older frames are already replaced by a text
                    # stub (see llm.prepare_for_model).
                    pair = outcome.get("images") or {}
                    for cam in ("agent", "wrist", ZOOM_KEY):
                        frame = pair.get(cam)
                        if frame is not None:
                            content.append({"type": "image", "image": frame})
                    if not content and outcome.get("image") is not None:
                        content.append({"type": "image", "image": outcome["image"]})

                # Deleted 2026-10-10: the grip-slip TREND tracker and its
                # "⛔ 物体掉了" event (`_prev_gap` + `_hold_gap`).
                #
                # It was built because "the model cannot see the trend by itself --
                # history is capped at MAX_HISTORY_MESSAGES, so it only ever sees
                # the most recent number". With 32 messages of history the model
                # can compare this turn's `爪上：` line with the previous one
                # itself, and `format_state` still reports the raw fact
                # (爪上：空 / 爪上：<物体>（离爪心 Xcm）) every single turn, so
                # nothing becomes unobservable.
                #
                # What it cost: a DELIBERATE `open` measured as a lost payload
                # (`open` clears `_grasped` -> gap None -> `_prev_gap` still set),
                # so the harness answered "物体掉了" to a model that had just put
                # the can down on purpose -- and the model spent the remaining 70
                # turns re-grasping it instead of walking to the basket
                # (MEASURED 2026-10-10; v4 in docs/P0_stripdown.md died the same
                # way). Cross-turn inference that the model can do itself is a
                # liability, not a safety net.
                content.append({
                    "type": "text",
                    "text": (
                        outcome["text"] + "\n"
                        + json.dumps(
                            model_view(outcome["result"], self.observation_mode),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n" + format_state(obs, self._grasped)
                    ),
                })
                messages.append({"role": "user", "content": content})
                transcript.append({
                    "turn": turn,
                    "result": outcome["text"],
                    "structured_result": outcome["result"],
                })
                if sim_done:
                    result.success = True
                    result.success_source = "env"
                    self._say(f"[episode] 环境判定成功（第 {turn} 轮，物体已放好），提前结束")
                    break

        except Exception as exc:  # noqa: BLE001 - keep episode artifacts even on crash
            result.error = result.error or f"episode failed: {type(exc).__name__}: {exc}"
            self._say(f"[episode] FATAL: {result.error}")
        finally:
            # Transcript + video are written on EVERY exit path (success,
            # failure, OOM, crash) — that is the point of the recorder.
            result.elapsed_s = time.time() - t_start
            result.transcript_path = str(self._write_transcript(episode_id, result, transcript))
            if recorder is not None:
                recorder.finalize(
                    success=result.success,
                    done_declared=result.done_declared,
                    turns=result.turns,
                    elapsed_s=result.elapsed_s,
                    error=result.error,
                )
        return result

    def _dispatch(self, call: ToolCall, obs: dict[str, Any]) -> dict[str, Any]:
        name, args = call.name, call.args
        original_name = name
        canonical = canonical_name(name)
        # Only `aim` is allowed to keep the "same pixel as last time" memory: any
        # other tool can have moved the arm, the object or the frame since.
        if canonical != "aim":
            self._last_aim = None
        if canonical == "observe":
            if args.get("target") or original_name == "locate":
                return {"obs": obs, "text": self._locate(args.get("target", args.get("name", "")), obs), "image": None, "done": False}
            if original_name == "look" or args.get("camera"):
                resp = self.client.look(camera=args.get("camera", "agent"))
                images = resp.images(self.client.ipc_dir)
                cam = args.get("camera", "agent")
                return {
                    "obs": resp.obs,
                    "text": f"[observe] refreshed {cam} view",
                    "image": images.get(cam),
                    "images": images,
                    "done": False,
                }
            return {"obs": obs, "text": f"[observe] {format_state(obs, self._grasped)}", "image": None, "done": False}
        if canonical == "approach":
            if original_name == "align_xy":
                name = "align_xy"
            else:
                name = "descend_to"
                args = {
                    **args,
                    "name": args.get("target", args.get("name", "")),
                    "z_offset_cm": args.get("height_offset_cm", args.get("z_offset_cm", "1.2")),
                }
        elif canonical == "gripper":
            name = "set_gripper"
        elif canonical in ("grasp", "release"):
            name = canonical
        elif canonical in ("aim", "target"):
            # `goto_pixel` (the pre-v2 name) lands here too: ALIASES maps it to
            # `target`, so old transcripts and old prompts keep working.
            name = canonical
        elif canonical == "verify":
            name = "declare_done"
            args = {**args, "answer_note": args.get("claim", args.get("answer_note", ""))}

        if name == "look":
            resp = self.client.look(camera=args.get("camera", "agent"))
            images = resp.images(self.client.ipc_dir)
            cam = args.get("camera", "agent")
            return {
                "obs": resp.obs,
                "text": f"[look {cam}] 已获取最新{cam}视角画面。",
                "image": images.get(cam),
                "images": images,
                "done": False,
            }

        if name == "grasp" or name == "release":
            # Concept primitives (档 A): the adapter owns the whole sequence
            # (open first, close to contact, squeeze, hold window / settle
            # window) and answers with a status from a closed vocabulary. The
            # harness passes that through and keeps the numbers in the
            # structured result -- the model reads the concept.
            resp = self.client.act({"kind": name})
            outcome = resp.outcome
            images = resp.images(self.client.ipc_dir)
            return {
                "obs": resp.obs,
                "text": format_outcome(outcome),
                "image": (images or {}).get("wrist"),
                "images": images,
                "done": False,
                "tool": name,
                "kind": outcome.get("kind", name),
                "note": f"{outcome.get('status', '')}",
                "status": outcome.get("status", ""),
                "held": outcome.get("held"),
                "measurement": outcome.get("measurement") or {},
                "requested": outcome.get("requested") or {},
            }

        if name == "aim":
            # 档 B v2 dry run: ground the pixel, draw the ring, DO NOT MOVE. The
            # harness never sees an object position on this path either -- the
            # simulator answers with the model's OWN pixel plus a concept.
            payload: dict[str, Any] = {
                "kind": "aim",
                "camera": args.get("camera", "agent"),
                "u": as_float(args.get("u"), 0.0),
                "v": as_float(args.get("v"), 0.0),
            }
            resp = self.client.act(payload)
            outcome = resp.outcome
            images = resp.images(self.client.ipc_dir)
            text = format_outcome(outcome)
            # Re-aiming the identical pixel with no motion in between cannot produce
            # new evidence (the frame and the ring are both unchanged), so say so
            # instead of letting the model pay a turn to rediscover it.
            key = (str(payload["camera"]), round(payload["u"]), round(payload["v"]))
            if self._last_aim == key:
                text += "（和上次 aim 完全相同，画面没有变化）"
            self._last_aim = key
            return {
                "obs": resp.obs,
                "text": text,
                "image": (images or {}).get(payload["camera"]),
                "images": images,
                "done": False,
                "tool": name,
                "kind": outcome.get("kind", name),
                "note": str(outcome.get("note", "")),
                "status": outcome.get("status", ""),
                "measurement": outcome.get("measurement") or {},
                "requested": outcome.get("requested") or {},
            }

        if name == "target":
            # 档 B: the model names a PIXEL; the simulator grounds it and servos a
            # SHORT BLOCK (`max_steps`, default 3, max 8). The model must look again
            # between blocks -- which costs nothing extra, since every action already
            # brings fresh frames back (see `attach_images` above).
            payload = {
                "kind": "target",
                "camera": args.get("camera", "agent"),
                "u": as_float(args.get("u"), 0.0),
                "v": as_float(args.get("v"), 0.0),
                "offset_mm": parse_vec3(args.get("offset_mm")),
            }
            if str(args.get("max_steps", "")).strip():
                payload["max_steps"] = as_float(args.get("max_steps"), 0.0)
            resp = self.client.act(payload)
            outcome = resp.outcome
            images = resp.images(self.client.ipc_dir)
            self._last_aim = None
            return {
                "obs": resp.obs,
                "text": format_outcome(outcome),
                "image": (images or {}).get("agent"),
                "images": images,
                "done": False,
                "tool": name,
                "kind": outcome.get("kind", name),
                "note": str(outcome.get("note", "")),
                "status": outcome.get("status", ""),
                "measurement": outcome.get("measurement") or {},
                "requested": outcome.get("requested") or {},
            }

        if name == "descend_to":
            # Aligned descent: keeps xy corrected the whole way down (measured
            # necessity — separate -z moves block at the can rim and close air).
            object_name = str(args.get("name", "")).strip()
            z_offset = as_float(args.get("z_offset_cm"), 1.2)
            resp = self.client.act(
                {"kind": "descend", "name": object_name, "z_offset_cm": z_offset}
            )
            outcome = resp.outcome
            images = resp.images(self.client.ipc_dir)
            end = outcome.get("eef_end", [0, 0, 0])
            note = str(outcome.get("note", ""))
            # Did it actually get DOWN to the object? MEASURED (v10, t28-t94,
            # 71 turns): the pads stopped 6.6 cm above the can
            # (`xy err 1.8 mm, z err 65.9 mm`) because that point is outside the
            # arm's reach -- probe_reach_map shows the same table reaching
            # z=0.009 at some cells and only z=0.16 at cells 4 cm away. The old
            # text still read "已对准并下降到 … 下一步用 set_gripper 操作夹爪",
            # so the model dutifully closed on air 71 times in a row.
            z_err = _z_err(note)
            # A stalled servo ("stuck at (x, y, z)") does NOT carry a `z err`, so
            # the old test let it fall through to the SUCCESS branch: measured
            # 2026-10-10 t69-t80, the result read "已对准并下降到
            # (-1.0392, 0.4899, 0.0122)" while the arm was a metre outside the
            # table and every following `close` came back on air. A stall is a
            # shortfall, exactly like a z error.
            stalled = "stuck" in note
            # Shortfall bar, kept EQUAL to the adapter's DESCEND_Z_TOL_M (5 mm).
            #
            # It was 20 mm, and the gap broke a whole episode: the descent pressed
            # on the can top for 565 ticks and stopped 12 mm high, which fell UNDER
            # the old 20 mm bar, so the model was told "已对准并下降到 …（z err
            # 12.0 mm）", believed it was down, closed there (finger width 0.0183 =
            # gripping the can's rim, vs 0.0314 when a descent converges) and shed
            # the can during the second +y carry -- after which 26 closes came back
            # on air (episode_000_1791632189). The verdict and the servo's own
            # tolerance must be the same number, or "arrived" means nothing.
            if stalled or (z_err is not None and z_err > DESCEND_SHORTFALL_M):
                detail = (
                    f"卡住了（{note}）"
                    if stalled
                    else f"终点 z={end[2]:.4f}，离目标还差 {z_err * 100:.1f}cm"
                )
                return {
                    "obs": resp.obs,
                    "text": f"[descend_to {object_name} +{z_offset}cm] 未到达：xy 已对准，{detail}",
                    "image": images.get("agent"),
                    "images": images,
                    "requested": outcome.get("requested", {}),
                    "raw": outcome,
                    "done": False,
                }
            return {
                "obs": resp.obs,
                "text": (
                    f"[descend_to {object_name} +{z_offset}cm] 已对准并下降到 "
                    f"({end[0]:.4f}, {end[1]:.4f}, {end[2]:.4f})（{note}，"
                    f"{outcome.get('steps', 0)} ticks）。"
                ),
                # `kind` still travels with the result for `format_outcome`, but
                # nothing in the episode loop keys off it any more: the self-harm
                # checks dispatch on the CANONICAL tool name (see `_tool`).
                "kind": outcome.get("kind"),
                "image": images.get("agent"),
                "images": images,
                "requested": outcome.get("requested", {}),
                "raw": outcome,
                "done": False,
            }

        if name == "align_xy":
            # Server-side primitive: precise xy alignment over an object, which
            # the model cannot do by arithmetic (measured: it closed on air with
            # 4.6 cm of x error). The model chooses what and when.
            object_name = str(args.get("name", "")).strip()
            resp = self.client.act({"kind": "align", "name": object_name})
            outcome = resp.outcome
            images = resp.images(self.client.ipc_dir)
            end = outcome.get("eef_end", [0, 0, 0])
            return {
                "obs": resp.obs,
                "text": (
                    f"[align_xy {object_name}] 已把机械爪 x/y 对准物体正上方（{outcome.get('note', '')}）；"
                    f"当前位置 ({end[0]:.4f}, {end[1]:.4f}, {end[2]:.4f})，用时 {outcome.get('steps', 0)} ticks。"
                ),
                "image": images.get("agent"),
                "images": images,
                "done": False,
            }

        if name == "locate":
            return {"obs": obs, "text": self._locate(args.get("name", ""), obs), "image": None, "done": False}

        if name == "get_state":
            return {"obs": obs, "text": f"[get_state] {format_state(obs, self._grasped)}", "image": None, "done": False}

        if name in ("move", "rotate", "set_gripper"):
            if name == "move":
                # 档 D: the model reports an AXIS and either a rung or a PIXEL GAP; the
                # code owns the centimetres. A pixel gap is the input it is actually good
                # at producing (its quoted gap runs 1.2-3.2x of truth) and the conversion
                # is the one it demonstrably fails at (~11x, mm-vs-cm). So: measure in
                # pixels, snap in log space, move in centimetres.
                ladder_cm = C.move_rung_cm(args.get("size", "")) if self.move_ladder else None
                gap_rung = None
                if self.move_ladder and ladder_cm is None:
                    gap_px = as_float(args.get("gap_px"), None)
                    if gap_px:
                        gap_rung, ladder_cm = C.rung_for_gap_cm(
                            gap_px * self.mm_per_px / 10.0, mm_per_px=self.mm_per_px
                        )
                action = {
                    "kind": "move",
                    "direction": args["direction"],
                    "distance_cm": (
                        ladder_cm if ladder_cm is not None
                        else as_float(args.get("distance_cm"))
                    ),
                }
                self._pending_gap = (args.get("gap_px"), gap_rung)
            elif name == "rotate":
                action = {
                    "kind": "rotate",
                    "axis": args["axis"],
                    "angle_deg": as_float(args.get("angle_deg")),
                }
            else:
                action = {"kind": "gripper", "action": args["action"]}
            resp = self.client.act(action)
            outcome = resp.outcome
            # 档 D instrument: label the step with the rung the model chose. The server
            # never sees this field -- it is the agent's own note on its own action, so
            # an audit can answer "which rung did it take, and did the rung do what it
            # means" without re-deriving it from the distance. Without it the only trace
            # of the ladder in a transcript would be a 10.0 that could also have been
            # typed by hand.
            if name == "move" and self.move_ladder:
                outcome["ladder_size"] = str(args.get("size", "")).strip().lower() or None
                gap_px, gap_rung = getattr(self, "_pending_gap", (None, None))
                if gap_rung:
                    outcome["ladder_size"] = gap_rung
                    outcome["gap_px"] = as_float(args.get("gap_px"), None)
            # Self-calibrate the pixel scale from the simulator's own measurement.
            mm_px = (outcome.get("measurement") or {}).get("mm_per_px")
            if mm_px:
                self.mm_per_px = float(mm_px)
            images = resp.images(self.client.ipc_dir)
            return {
                "obs": resp.obs,
                "text": format_outcome(outcome),
                "image": images.get("agent"),
                "images": images,
                "done": False,
                # Keep the raw outcome fields the episode loop needs in order to
                # track grasp state. Without them `_grasped` NEVER became True:
                # the loop's grasp test was always seeing None, so the "它目前在
                # 夹爪里" fact never appeared in `locate` and the model could not
                # tell it was already holding the can. P0c: it re-grasped 11 times
                # and pushed the can 38 cm away.
                # NOTE: these two `kind` fields are for `format_outcome` only.
                # The episode loop now dispatches on `outcome["tool"]` (the tool
                # name) -- see the note in `run_episode`: the adapter's kind string
                # is not the same vocabulary and comparing against it silently
                # disabled a warning for two whole runs.
                "kind": outcome.get("kind"),
                "note": outcome.get("note", ""),
                "requested": outcome.get("requested", {}),
                # `status` / `held` / `measurement` are carried for the SAME reason the
                # grasp branch carries them: `raw` mode's `gripper` has to be judged
                # from the contact table (see the episode-loop note), and
                # `format_outcome` reads the concept this way. For `move`/`rotate` the
                # adapter leaves them empty, so this is additive.
                "status": outcome.get("status", ""),
                "held": outcome.get("held"),
                "measurement": outcome.get("measurement") or {},
                # Keep the adapter's own dict so the v2 envelope can carry the
                # MEASURED numbers. `format_outcome` consumes them to build the
                # human text, and without this copy the structured `measurement`
                # shipped empty -- measured 2026-10-10: `approach` returned
                # `"measurement": {}` on 27/27 calls, so the "返回实测位移" the
                # v2 contract advertises never reached the model.
                "raw": outcome,
            }

        if name == "declare_done":
            success = self.client.success()
            if success:
                text = f"[declare_done] 验证通过，任务完成！说明：{args.get('answer_note', '')}"
            else:
                text = "[declare_done] 任务完成检查**未通过**：目标物体还没有按要求放好。"
            return {"obs": obs, "text": text, "image": None, "done": True, "success": success}

        if name == "write_skill":
            skill_name = _safe_skill_name(args.get("name", "untitled"))
            path = self.skills_dir / f"{skill_name}.md"
            path.write_text(str(args.get("content", "")).strip() + "\n")
            return {
                "obs": obs,
                "text": f"[write_skill] 已保存技能 “{skill_name}”。",
                "image": None,
                "done": False,
            }

        if name == "read_skill":
            skill_name = _safe_skill_name(args.get("name", ""))
            path = self.skills_dir / f"{skill_name}.md"
            if path.exists():
                text = f"[read_skill {skill_name}]\n{path.read_text()}"
            else:
                available = sorted(p.stem for p in self.skills_dir.glob("*.md"))
                text = f"[read_skill] 没有名为 {skill_name} 的技能。现有技能: {available or '（空）'}"
            return {"obs": obs, "text": text, "image": None, "done": False}

        raise ValueError(f"unknown tool: {name!r}")

    # ------------------------------------------------------------------ util
    def _structured_outcome(self, call: ToolCall, outcome: dict[str, Any]) -> dict[str, Any]:
        # Read the numbers from the ADAPTER's dict, not from the flattened result
        # `_dispatch` builds for the model. The flat dict never had eef_start /
        # eef_end / measured_delta_m / steps at its top level, so the lookup below
        # silently produced `"measurement": {}` for every motion call.
        raw = outcome.get("raw") or outcome.get("outcome") or outcome
        measurement = {
            key: raw[key]
            for key in ("eef_start", "eef_end", "measured_delta_m", "steps")
            if raw.get(key) is not None
        }
        status = "verified" if outcome.get("done") and outcome.get("success") else (
            "not_verified" if outcome.get("done") else "completed"
        )
        # Concept primitives (档 A) answer with their own closed vocabulary
        # (`grasped`/`empty`/`slipped`/`released`/`still_carrying`) and carry the
        # numbers separately. Both belong in the transcript: the model reads the
        # concept, while the offline audits (release poses, slip counts) need
        # `contact_sides` / `object_body` / `width_mm` at full resolution.
        concept_status = str(outcome.get("status") or "")
        if concept_status:
            status = concept_status
        measurement.update(outcome.get("measurement") or {})
        # 档 D instrument: keep the RUNG in the transcript, both as the word the model
        # picked and as the centimetres code turned it into. An audit that only saw
        # `distance_cm: 10.0` could not tell a ladder step from a hand-typed one, and
        # "did the rung mean what it says" is the question the experiment asks.
        #
        # THREE places, and this cost a debugging round: `_dispatch` annotates the
        # ADAPTER outcome (which is what it hands to `format_outcome`), and ships it here
        # as `raw`. Reading only the top level silently recorded `move_size: null` for
        # exactly the calls whose rung was DERIVED from a pixel gap -- the new feature --
        # while the model-visible text correctly said "size=far".
        raw_obj = outcome.get("raw")
        raw_rung = (
            raw_obj.get("ladder_size") if isinstance(raw_obj, dict)
            else getattr(raw_obj, "ladder_size", None)
        )
        rung = outcome.get("ladder_size") or raw_rung or call.args.get("size")
        if rung and canonical_name(call.name) == "move":
            measurement["move_size"] = str(rung).strip().lower()
            measurement["move_size_cm"] = C.move_rung_cm(str(rung))
        if canonical_name(call.name) == "move" and call.args.get("gap_px"):
            measurement["gap_px"] = float(as_float(str(call.args["gap_px"]), 0.0))
        events: list[dict[str, Any]] = []
        # A motion that fell short is an EVENT, and both surfaces must say so with
        # the same vocabulary: `move` reports "只走了 Xcm" and `descend_to`
        # reports "未到达". Only the first was tested, so `approach` could come
        # back with an empty event list while its own text said it never got down
        # to the object.
        text = str(outcome.get("text", ""))
        if "只走了" in text or "未到达" in text:
            status = "partial"
            events.append({"type": "motion_shortfall"})
        if outcome.get("done") and not outcome.get("success"):
            events.append({"type": "completion_check_failed"})
        return structured_result(
            tool=canonical_name(call.name),
            status=status,
            requested=outcome.get("requested") or raw.get("requested") or dict(call.args),
            measurement=measurement,
            events=events,
        )

    def _locate(self, query: str, obs: dict[str, Any]) -> str:
        """Report a GT object position (the 'object detector' affordance).

        Added to M1 after the first real episode: the model descended to z=0.20
        and closed its fingers in mid-air because pixel input alone gives it no
        depth. `locate` hands it the same coordinates the dry-run policy used,
        turning a perception problem into the geometric execution the adapter
        is actually good at. The vision-only comparison stays available as an
        ablation (`--no-images` is the other half of that pair).
        """
        objects: dict[str, Any] = obs.get("objects", {})
        if not objects:
            return "[locate] 当前没有物体信息（服务器未提供）。"

        query_norm = str(query).strip().lower().replace(" ", "_")
        match = None
        if query_norm in objects:
            match = query_norm
        else:
            for key in objects:
                stem = key.rsplit("_", 1)[0].lower()
                if query_norm and (query_norm in key.lower() or query_norm in stem or stem in query_norm):
                    match = key
                    break

        if match is None:
            names = ", ".join(sorted(objects))
            return f"[locate] 未找到与“{query}”匹配的物体。可用物体: {names}"

        pos = objects[match]
        eef = obs.get("eef_pos", [0.0, 0.0, 0.0])
        dx, dy, dz = pos[0] - eef[0], pos[1] - eef[1], pos[2] - eef[2]

        # P0b: the `hint()` helper that translated a deviation into a ready-made
        # move command is gone. It is what told the model "x 轴已对齐；y 轴已对齐"
        # while dz was still -5.5cm (P0a' t11) -- the model obeyed it and never
        # descended. Turning measurements into commands is the model's job.
        hold_gap = float(
            np.linalg.norm(np.asarray(pos, dtype=float) - np.asarray(eef, dtype=float))
        )
        # Decide FIRST whether the query is about the object already in the pads,
        # because the two cases must give MUTUALLY EXCLUSIVE advice.
        #
        # v9 showed exactly why (t62-t80): the reply contained BOTH
        # "下一步建议: 先 move +x 1.6cm" -- computed from the object riding the
        # gripper, i.e. pure noise -- AND a trailing paragraph saying to ignore
        # those numbers. The model followed the FIRST one and ran `move -y 15`
        # (t64/t65) straight away from the basket while the can was in its hand.
        # Two pieces of advice that contradict each other one line apart are
        # worse than none. Same rule as `format_state`'s 爪上： line, so the two
        # signals can never disagree.
        holding = (
            getattr(self, "_grasped", False)
            and _carried_name(obs) == match
            and hold_gap < 0.06
        )

        if holding:
            # Height gate. MEASURED (v9/v10/v11 -- bit-identical because
            # temperature=0): the can was carried at centre z=0.126-0.152, i.e.
            # BOTTOM at 0.072-0.098, while the basket rim sits at z~=0.115. The
            # final `move +y 15` (t19) swept the payload straight across the rim
            # and scraped it off at the same spot in all three runs -- 11 cm
            # short of the basket, which made it look like "it keeps dropping
            # near the basket" rather than "it hits the basket".
            # The tool result has to say whether the height is legal, because the
            # model cannot see the rim in the numbers.
            lines = [
                f"[locate] {match}: ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m",
                f"  机械爪当前: ({eef[0]:+.4f}, {eef[1]:+.4f}, {eef[2]:+.4f}) m",
                # P0b: state the FACT and stop. The old wording continued with
                # "不要按它们的差值移动…也不要再 descend_to / close" and then
                # prescribed a scene-specific height rule (rim 0.115, centre
                # >=0.17). Facts belong in the result; prescriptions do not.
                f"  （它目前在夹爪里，离爪心 {hold_gap * 100:.1f}cm）",
            ]
            # P0b: no height gate, no slip threshold, no next-step recipe.
            # All three were scene constants I had fitted to past crashes
            # (rim 0.115 / centre 0.17 / 3.8cm) and all three prescribe action.
        else:
            # P0b: measurements ONLY.
            #
            # What the old text appended, and what it cost (P0a', t9-t24):
            #   "下一步建议: 先 move +x 0.3cm；再 y 轴已对齐"
            #   "抓取高度: x/y 对齐后降到 z≈0.050 m"
            # With the pads 5.5 cm ABOVE the can (dz=-5.5cm) the advice line
            # announced "x 轴已对齐；y 轴已对齐", so the model believed alignment
            # was finished and spent 12 turns nudging x/y back and forth (t12-t23)
            # instead of descending -- then its next descent slid 11 cm sideways
            # and shoved the can away. The advice never once mentioned the axis
            # that was actually wrong. A hint that silently ignores the one
            # mismatched dimension is worse than no hint.
            lines = [
                f"[locate] {match}: ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m",
                f"  机械爪当前: ({eef[0]:+.4f}, {eef[1]:+.4f}, {eef[2]:+.4f}) m",
                f"  偏差（物体 - 机械爪）: dx={dx * 100:+.1f}cm, dy={dy * 100:+.1f}cm, dz={dz * 100:+.1f}cm",
            ]
        # Deleted 2026-10-10: the "附近还有 <物体>，距它 Xcm" proximity warning.
        # It compared the queried object against every OTHER object in the scene
        # and reported pairs closer than a fitted 7 cm. The model can ask for any
        # object's coordinates itself (`observe` with that target), so this was
        # the harness doing a second query and a distance subtraction on its
        # behalf -- i.e. the same "derive what the model can derive" pattern as
        # the slip/hurt trackers, with a scene constant (0.07 m) thrown in.
        return "\n".join(lines)

    def _skills_index(self) -> str:
        # The skill tools (write_skill / read_skill) exist ONLY on the legacy
        # surface. Under v2 this text was still appended to the system prompt of
        # every turn, inviting the model to call tools that are not in its
        # schema -- a dangling instruction to an unknown tool.
        if not self.legacy_tools:
            return ""
        skills = sorted(p.stem for p in self.skills_dir.glob("*.md"))
        if not skills:
            return "\n\n## 技能库\n（当前为空——如果你总结出有用的经验，可以用 write_skill 记录。）"
        lines = "\n".join(f"- {name}" for name in skills)
        return f"\n\n## 技能库（可用 read_skill 读取）\n{lines}"

    def _write_transcript(
        self, episode_id: int, result: EpisodeResult, transcript: list[dict[str, Any]]
    ) -> Path:
        out_dir = self.workspace / "outputs" / "episodes"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"episode_{episode_id:03d}_{int(time.time())}.json"
        payload = {
            "result": {
                "task": result.task,
                "language": result.language,
                "success": result.success,
                "success_source": result.success_source,
                "done_declared": result.done_declared,
                "turns": result.turns,
                "tool_calls": result.tool_calls,
                "protocol_retries": result.protocol_retries,
                "elapsed_s": round(result.elapsed_s, 1),
            },
            "transcript": transcript,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        return path

    def _say(self, message: str) -> None:
        if self.log:
            print(message, flush=True)


# ------------------------------------------------------------------ format
# Positions are printed to 0.1 mm (4 decimals). 3 decimals (1 mm) quantises the
# feedback enough to trap a 2 mm-tolerance controller in a limit cycle: the
# dry-run policy read "-0.119" , moved <1 mm, and read the same rounded value
# again, so it never converged. The LLM needs the same sub-mm resolution.
# MEASURED finger widths (robosuite Panda). Kept in sync with
# action_adapter.GRIPPER_OPEN_WIDTH / GRIPPER_EMPTY_WIDTH -- duplicated rather
# than imported so the agent side never pulls in a MuJoCo-importing module.
GRIPPER_OPEN_WIDTH = 0.034  # at/above this the pads are still open (no grasp)
GRIPPER_EMPTY_WIDTH = 0.006  # at/below this they are fully shut on air
# (P0b removed BASKET_RIM_Z / CARRY_SAFE_CENTRE_Z: scene constants fitted to
# past crashes, no longer referenced by any code path.)


def _fmt3(vec: Any) -> str:
    return f"({vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f})"


# Keys that identify a SCENE OBJECT, or hand over a world POSITION, rather than
# describe what happened. They stay in the transcript (the offline audits read
# `measurement` at full resolution) but are dropped from the copy the model is shown
# when the observation mode claims not to know about objects or coordinates.
#
# 档 B v2 added the coordinate keys. Until then `target`/`goto_pixel` results carried
# `surface_xyz` (the grounded point), `target_xyz` (where the arm was sent) and
# `offset_m` -- i.e. the model was handed the world position it had just asked for.
# It is not scene truth (a depth camera would give the surface, and the offset is the
# model's own input), but it is exactly the numeric form this mode exists to remove:
# the model is supposed to decide from the ring in the picture, and 10-13 cm of
# measured pointing bias says it reads POSITIONS badly anyway. `mm_per_px` deliberately
# stays model-facing (it is the scale it needs to convert cells into centimetres).
MODEL_HIDDEN_KEYS = (
    "object_body",
    "surface_xyz",
    "target_xyz",
    "offset_m",
    "depth_m",
    "mark",
)


def model_view(result: dict[str, Any], observation_mode: str) -> dict[str, Any]:
    """The structured result as the MODEL sees it (档 B).

    HISTORY, because this is a leak that was found by a test rather than by
    reading: every tool result carries its structured JSON, and 档 A's grasp concept
    put the contact-derived body name in it (`measurement.object_body =
    "alphabet_soup_1_main"`). In `pixels` mode that hands the model the simulator's
    object identity on a path whose whole point is that it has none.
    """
    if str(observation_mode).strip().lower() != "pixels":
        return result
    view = dict(result)
    measurement = dict(view.get("measurement") or {})
    for key in MODEL_HIDDEN_KEYS:
        measurement.pop(key, None)
    if measurement:
        view["measurement"] = measurement
    else:
        view.pop("measurement", None)
    return view


def parse_vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[float]:
    """Parse a 3-vector the model may hand over in any of the shapes seen.

    The tool arguments arrive as STRINGS (they come out of the XML/JSON tool call),
    so `offset_mm` shows up as `"[0, 0, 120]"`, `"0,0,120"` or `"0 0 120"` about
    equally often. MEASURED: without this the harness iterated the characters of the
    string, sent 14 numbers, and the server rejected the call -- the model saw
    "offset_mm 需要三个数" for an argument that looks correct in its own message.
    """
    if value is None or value == "":
        return list(default)
    if isinstance(value, (list, tuple)):
        items: list[Any] = list(value)
    else:
        text = str(value).strip().strip("[](){} ")
        items = [chunk for chunk in re.split(r"[,\s]+", text) if chunk]
    numbers: list[float] = []
    for item in items:
        try:
            numbers.append(float(str(item).strip()))
        except (TypeError, ValueError):
            continue
    if len(numbers) != 3:
        return list(default)
    return numbers


def _z_err(note: str) -> float | None:
    """Pull 'z err 65.9 mm' out of an adapter note and return it in metres."""
    m = re.search(r"z err ([-0-9.]+) mm", note or "")
    return float(m.group(1)) / 1000.0 if m else None


def _object_key_from_body(body: str | None) -> str | None:
    """Sim body name -> observation object key.

    The adapter reads names out of the contact table, where objects appear with
    their MJCF body name (`alphabet_soup_1_main`), while `obs["objects"]` is keyed
    by the `<name>_pos` observable (`alphabet_soup_1`). One place converts, so the
    release instrument compares like with like.
    """
    if not body:
        return None
    text = str(body)
    return text[: -len("_main")] if text.endswith("_main") else text


def _carried_name(obs: dict[str, Any], tol_m: float = 0.06) -> str | None:
    """Which object is currently between the pads (nearest to the EEF).

    This is what lets the model NOTICE a dropped payload on its own: after every
    action it sees whether anything is still riding on the gripper. Measured
    (probe_move_chunking / probe_carry_order): a carry sheds the can on a small
    fraction of moves, and the model is expected to spot that and re-grasp -- so
    it must be given the signal.
    """
    eef = obs.get("eef_pos")
    objects = obs.get("objects") or {}
    if eef is None or not objects:
        return None
    best, best_d = None, 1e9
    for name, pos in objects.items():
        d = float(np.linalg.norm(np.asarray(pos, dtype=float) - np.asarray(eef, dtype=float)))
        if d < best_d:
            best, best_d = name, d
    return best if best_d < tol_m else None


def format_state(obs: dict[str, Any], grasped: bool = False) -> str:
    """One line of state: where the gripper is, and whether it holds anything.

    CHANGED 2026-10-11 (档 A): the gripper half is a CONCEPT now, and its source
    moved from the finger width to the CONTACT table (the caller passes the
    adapter's `held`). MEASURED at reset the pads sit half closed (0.0207) with
    zero contacts, so the old `张开 / 闭合（空）/ 半闭（指间有物体）` classification
    called an empty hand "指间有物体" at the start of every episode.

    The object NAME and its distance to the gripper centre are gone too: the
    name came from ground-truth object positions (`_carried_name`), i.e. the
    harness was leaking scene truth through a string the model reads. What the
    model needs to notice a drop is *whether* something is between the pads, and
    that is exactly what is left. The identity and the millimetres stay in the
    transcript (`measurement.object_body`, `width_mm`).
    """
    eef = obs.get("eef_pos", [0, 0, 0])
    hold = "爪上：有物体（两指接触）" if grasped else "爪上：空"
    return f"机械爪位置 {_fmt3(eef)} m；{hold}"


def moved_m(outcome: dict[str, Any]) -> float:
    """How far the END EFFECTOR moved, in metres, from a dispatched outcome.

    Both sources are consulted on purpose, and getting this wrong has already cost one
    diagnosis: `_dispatch` copies the ADAPTER's `measurement` dict (which only the
    concept primitives fill), so for a plain `move` that dict is `{}` -- the numbers live
    on the adapter object itself, kept under `raw`. Reading only one of the two is how
    the stall guard came to call 8.3mm raises "no movement at all".
    """
    raw = outcome.get("raw")
    delta = getattr(raw, "measured_delta_m", None)
    if not delta:
        delta = (outcome.get("measurement") or {}).get("measured_delta_m")
    if not delta:
        delta = ((outcome.get("result") or {}).get("measurement") or {}).get("measured_delta_m")
    try:
        return max((abs(float(v)) for v in delta), default=0.0)
    except (TypeError, ValueError):
        return 0.0


def format_outcome(outcome: dict[str, Any]) -> str:
    kind = outcome.get("kind")
    requested = outcome.get("requested", {})
    delta = outcome.get("measured_delta_m", [0, 0, 0])
    end = outcome.get("eef_end", [0, 0, 0])
    ticks = outcome.get("steps", 0)
    if kind == "aim":
        # 档 B v2 dry run. The model gets back its OWN pixel, the concept of what
        # happened to it, and the local scale -- never a position, which is the whole
        # point of the mode (and `model_view` drops even the world point it asked for).
        camera = str((requested or {}).get("camera", ""))
        pixel = (requested or {}).get("pixel") or []
        where = f"像素({pixel[0]},{pixel[1]})" if len(pixel) == 2 else ""
        if str(outcome.get("status", "")) == "rejected":
            return (
                f"[aim {camera} {where}] 拒绝：{outcome.get('note', '')}"
                "（这张图里没有画红圈，换个像素再 aim）"
            )
        mm_px = (outcome.get("measurement") or {}).get("mm_per_px")
        scale = f"，此处 1 像素≈{mm_px}mm" if mm_px else ""
        return (
            f"[aim {camera} {where}] 红圈已画在你说的像素上（青色十字是夹爪自己）{scale}"
            "；另附一张围绕红圈的 4 倍放大图：看红圈中心是否在目标内部、四边留边是否均匀，"
            "满意就 target，不满意改像素再 aim"
        )
    if kind in ("target", "goto_pixel"):
        # The model gets back its OWN pixel and the local scale (mm/px) it needs to
        # plan the next click -- never the object's position, which is the whole
        # point of the mode.
        camera = str((requested or {}).get("camera", ""))
        pixel = (requested or {}).get("pixel") or []
        where = f"像素({pixel[0]},{pixel[1]})" if len(pixel) == 2 else ""
        mm_px = (outcome.get("measurement") or {}).get("mm_per_px")
        status = str(outcome.get("status", ""))
        if status == "rejected":
            return f"[{kind} {camera} {where}] 拒绝：{outcome.get('note', '')}（爪子没有动）"
        concept = {
            "arrived": "已到位",
            "short": "没到位（还差一点）",
            "blocked": "中途被挡住/到不了",
        }.get(status, "结果未知")
        scale = f"，此处 1 像素≈{mm_px}mm" if mm_px else ""
        # The magnifier is attached on every motion call as well, centred on the
        # CROSSHAIR: after a short block the useful question is "how far is my hand
        # from the object", and at 1x the answer is 4x harder to read.
        zoom_note = "；另附围绕夹爪十字的放大图" if status != "rejected" else ""
        # `truncated` is the SHORT BLOCK running out, which is a different fact from
        # "blocked": the arm is fine, the model just stopped looking. Say which one it
        # is, or a short block reads as a failure to be worked around.
        if status == "short" and (outcome.get("measurement") or {}).get("truncated"):
            return (
                f"[{kind} {camera} {where}] {concept}：这一小段走完还没到，"
                f"同一像素再 target 一次即可{scale}{zoom_note}"
            )
        return f"[{kind} {camera} {where}] {concept}{scale}{zoom_note}"
    if kind in ("grasp", "release"):
        # Concept primitives: the model is told WHAT happened, not how many
        # millimetres the fingers travelled. The numbers are in the structured
        # result (`measurement`), which is where the instruments read them.
        status = str(outcome.get("status", ""))
        concept = {
            "grasped": "夹住了（两指都接触物体，握力已加够）",
            "empty": "没夹到东西（指间是空的）",
            "slipped": "接触不稳定（只碰到一侧或握持中松开）",
            "released": "已放开（物体已离开指间）",
            "still_carrying": "还夹着（物体未脱离指间）",
        }.get(status, status or "结果未知")
        return f"[{kind}] {concept}"
    if kind == "move":
        # MEASURED (v10 t24/t33/t41/t47/t50): several `move -z` calls returned
        # 10-60% of the requested distance with a completely silent result --
        # the adapter's "blocked" note was being DROPPED here, so a stalled
        # descent looked exactly like a successful one. The model then closed
        # the gripper ~7 cm above the can, 71 turns in a row. A move that fell
        # far short of its request has to say so.
        note = str(outcome.get("note", ""))
        try:
            want = abs(float(requested.get("distance_cm") or 0.0)) / 100.0
        except (TypeError, ValueError):
            want = 0.0
        axis = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2}.get(
            str(requested.get("direction")), 0
        )
        got = abs(float(delta[axis]))
        rung = outcome.get("ladder_size")
        # With the ladder on, say the WORD as well as the centimetres: the model's
        # downshift decision is "was that rung too big", and a bare 10.0 asks it to map
        # a number back to the word it chose.
        rung_tag = f" size={rung}" if rung else ""
        text = (
            f"[move {requested.get('direction')}{rung_tag} {requested.get('distance_cm')}cm] "
            f"实测位移 {_fmt3(delta)} m（{ticks} ticks）；机械爪现在位于 {_fmt3(end)}"
        )
        # THE RULER, measured on the model's OWN step. Three arms measured the same defect:
        # its "I am still N cm away" is 2.4-6x too small (it converts the global view at
        # ~1mm/px where the camera measures 3.0-4.0mm/px), so every step it picks is too
        # small -- 54-72% of quoted moves underestimate by 2x or more. Telling it the scale
        # is not enough (the prompt already states it, and it is still wrong); this line lets
        # it calibrate against something it cannot argue with: the step it just took.
        px = (outcome.get("measurement") or {}).get("delta_px")
        mm_px = (outcome.get("measurement") or {}).get("mm_per_px")
        if px and mm_px:
            text += (
                f"；这一步在全局画面里让爪心移动了 {px[0]:+.1f} / {px[1]:+.1f} 像素"
                f"（此处 1 像素≈{mm_px:.1f}mm）"
            )
        if want > 0.005 and got < want * 0.5:
            # Fact (the shortfall) only. The old text added "don't repeat the
            # same direction; try another, or lift first" -- that is planning.
            text += f" ⚠ 只走了 {got * 100:.1f}cm（要求 {want * 100:.1f}cm）"
            if note:
                text += f" [{note}]"
        return text
    if kind == "rotate":
        note = outcome.get("note", "")
        return (
            f"[rotate {requested.get('axis')} {requested.get('angle_deg')}°] {note}（{ticks} ticks）；"
            f"机械爪位于 {_fmt3(end)}"
        )
    if kind == "set_gripper":
        note = outcome.get("note", "")
        action = requested.get("action")
        # CONCEPT FIRST (`raw` mode). The contact-derived verdict is the one that can
        # be trusted: MEASURED, the width test calls an empty hand "holding" at reset
        # (pads start at 0.0207 with zero contacts) and cannot tell a can from a rim
        # (0.0183 vs 0.0197). The width sentence below is only the fallback for a
        # transcript whose adapter did not carry a verdict.
        concept = {
            "grasped": "夹住了（两指都碰到物体）",
            "empty": "夹空了（指间是空的）",
            "slipped": "接触不稳（只碰到一侧）",
            "released": "已张开（指间无物体）",
            "still_carrying": "张不开（还夹着东西）",
        }.get(str(outcome.get("status") or ""))
        if concept:
            return (
                f"[gripper {action}] {concept}（{ticks} ticks）；"
                f"机械爪位于 {_fmt3(end)}"
            )
        # Turn the raw finger width into the decision the model needs.
        #
        # BUG FIXED (this is what dropped the can twice in the 80-turn run):
        # the old test was a SUBSTRING match, `"width 0.0" in note`, which is
        # true for EVERY width below 0.1 -- 0.0229 and 0.0300 included. All 13
        # real grasps of that episode were reported as "夹空了", so the model
        # released the can and retried; the release is what actually dropped it.
        #
        # MEASURED finger widths (robosuite Panda, probe_action_semantics +
        # probe_align_descend): fully open ~0.0388, shut on air ~0.0007, and an
        # object between the pads parks them at ~0.018-0.030.
        verdict = ""
        if action == "close":
            width = _grip_width(note)
            if width is None:
                verdict = ""
            elif width <= GRIPPER_EMPTY_WIDTH:
                # Retry guidance matters: the old text said "下降更多后再试", and
                # the model implemented that as `descend_to(name, -0.3)`.
                # MEASURED (v6 t20): that negative-offset descend scraped the can
                # sideways and pushed it 5.4 cm -- straight toward the basket,
                # which was then knocked over at t21. Lift first, then re-align.
                verdict = " 夹空了（两指完全闭合）"
            elif width >= GRIPPER_OPEN_WIDTH:
                verdict = " 夹爪未闭合（仍在张开位置）"
            else:
                verdict = " 指间有物体（夹住了）"
        elif action == "open":
            # Tell the model what an open means RIGHT HERE, every time. The v4
            # episode failed because it read `爪上：空` after releasing the can
            # into the basket, concluded "it fell out", and spent 36 turns
            # chasing it. `爪上：空` is EXPECTED after a deliberate release.
            verdict = " 已张开（指间无物体）"
        return (
            f"[set_gripper {action}] {note}（{ticks} ticks）；"
            f"机械爪位于 {_fmt3(end)}{verdict}"
        )
    return f"[{kind}] {outcome}"


def _grip_width(note: str) -> float | None:
    """Pull the finger width out of an adapter note (numeric, never a substring)."""
    m = re.search(r"width ([0-9.]+)", note)
    return float(m.group(1)) if m else None


def _safe_skill_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(name).strip().lower())
    return cleaned.strip("-") or "untitled"


def _fmt_call(call: ToolCall) -> str:
    args = ", ".join(f"{k}={v}" for k, v in call.args.items())
    return f"{call.name}({args})"


def compact_history(messages: list[dict[str, Any]], max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict[str, Any]]:
    """Bound the transcript: keep the system prompt, the task message, and the
    newest turns; older turns collapse into one note.

    Rationale (measured): on this GPU attention has no memory-efficient kernel,
    so the cost of a turn grows with the square of the token count. The first
    real episode died at turn 17 with the allocator asking for exactly one
    N x N attention matrix. M2 replaces this mechanical trim with a
    model-written summary; M1 only needs the ceiling.
    """
    if len(messages) <= max_messages:
        return messages
    head = messages[:2]  # system + initial task message
    tail = messages[-(max_messages - 3) :]  # newest turns + room for the note
    dropped = len(messages) - len(head) - len(tail)
    # Never start the tail on an assistant turn without its user message.
    while tail and tail[0].get("role") != "user":
        tail = tail[1:]
        dropped += 1
    note = {
        "role": "user",
        "content": (
            f"[上下文压缩] 更早的 {dropped} 条消息已省略（为避免超出模型上下文）。"
            "如需要，可以重新观察当前状态。"
        ),
    }
    return [*head, note, *tail]
