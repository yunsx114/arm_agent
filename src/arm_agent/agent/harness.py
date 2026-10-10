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

from arm_agent.agent.llm import prepare_for_model
from arm_agent.agent.parser import (
    ToolCall, as_float, parse_tool_calls, truncate_to_first_call, validate_calls,
)
from arm_agent.agent.prompts import (
    get_system_prompt,
    get_tool_schemas,
    get_verifier_prompt,
    get_verifier_tools,
)
from arm_agent.agent.tool_protocol import canonical_name, structured_result
from arm_agent.agent.render import ACCENT, EpisodeRecorder, GREEN, RED, YELLOW

SKILLS_DIRNAME = "skills"
# Kept equal to `action_adapter.DESCEND_Z_TOL_M`: a descent that stops further
# than this from the commanded z did not arrive, and saying it did is what makes
# the model close on a rim instead of re-descending.
DESCEND_SHORTFALL_M = 0.005
# The shared skill file, injected into the executor AND the verifier prompts.
SHARED_SKILL_FILE = "scene_tools.md"
# Feed-back images: the newest N frames stay as pixels, the rest become text
# stubs. Was 1 while only the agent view was attached; the loop now attaches the
# agent AND wrist cameras each step, so 2 keeps exactly one whole frame pair
# (keeping 1 would silently throw away half of the newest pair). Tokens: 2 x
# 66 = 133 per turn, and only for the newest turn.
HISTORY_IMAGE_KEEP = 2
# The verifier replies with one `release` call and a one-sentence reason; with the
# reasoning block off it needs ~120 tokens. Kept small on purpose: the format rule
# for the verifier has no preamble, so anything larger only invites a monologue
# that could get truncated before the call closes.
VERIFIER_MAX_NEW_TOKENS = 512
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
        verifier: bool = False,
        verifier_thinking: bool | None = None,
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
        self.image_keep = int(image_keep) if image_keep is not None else HISTORY_IMAGE_KEEP
        # "full" = the accumulated scene-specific prompt; "plain" = task +
        # frame + tool semantics only (see prompts.get_system_prompt).
        self.prompt_mode = "v2" if tool_protocol == "v2" and prompt_mode == "full" else prompt_mode
        self.tool_protocol = tool_protocol
        self.tool_schemas = get_tool_schemas(tool_protocol)
        self.legacy_tools = str(tool_protocol).lower() != "v2"
        self.skills_dir = self.workspace / SKILLS_DIRNAME
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        # SHARED SKILL, injected into BOTH roles' system prompts.
        #
        # It lives on disk (not inline in prompts.py) so it can be edited /
        # self-iterated without touching code. Read once here: the system prompt
        # string is rebuilt every turn, and re-reading a file each turn would be
        # pure I/O overhead.
        #
        # Why both roles need it: the executor had the axis convention in its own
        # prompt while the verifier did not, and the verifier's picture-relative
        # hints ("篮子在右侧") made the executor march in +x while the real
        # deficit was +y -- 40/40 moves in the wrong axis, measured. One shared
        # vocabulary is the fix.
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
        # Two-role release gate. Same weights, DIFFERENT context: when the
        # executor asks to open the gripper while carrying a payload, the harness
        # suspends that context and asks a verifier (fresh message list, its own
        # system prompt, one tool: `release`) whether the action is safe.
        #
        # Why a separate role instead of a prompt rule: measured 2026-10-10, the
        # executor asserts arrival without checking (5/5 releases proceeded on a
        # false "已到达篮子正上方"; 0/80 turns doubted the release position), and
        # when the executor is asked the same question directly it answers
        # correctly (8/8 single-turn, 4/4 at full run context with the reasoning
        # block on, 3/4 with it off). The gate moves the question into the loop.
        # Only `gripper open` while `_grasped` is gated -- an `open` on an empty
        # hand is just part of a re-grasp cycle (22 of them in the last run).
        self.verifier_enabled = bool(verifier)
        self.verifier_thinking = verifier_thinking
        self.verifier_schemas = get_verifier_tools()
        self.verifier_prompt = get_verifier_prompt() + self.shared_skill
        self.verifier_calls = 0
        self._task_text = ""

    # ------------------------------------------------------------- verifier
    def _release_verdict(
        self, executor_reply: str, images: dict[str, Any] | None
    ) -> tuple[bool, str]:
        """Ask the verifier role whether the pending `gripper open` may proceed.

        Fresh context on purpose: the verifier sees the task, the two current
        camera frames and the executor's own words -- and NOT the executor's
        history, so it cannot inherit the executor's "already close enough"
        narrative. Same weights, so this costs one prefill plus a short reply.

        Fail-closed: a missing/garbled verdict rejects. Rejecting only costs one
        more move; releasing wrongly drops the payload and ends the attempt.
        """
        pair = images or {}
        content: list[dict[str, Any]] = []
        for cam in ("agent", "wrist"):
            frame = pair.get(cam)
            if frame is not None:
                content.append({"type": "image", "image": frame})
        content.append(
            {
                "type": "text",
                "text": (
                    f"任务目标：{self._task_text}\n"
                    "执行者现在请求执行的动作：张开夹爪（gripper open）放下物体。\n"
                    "以下是执行者这一轮的完整自述（可能不准，以画面为准）：\n"
                    "<执行者自述>\n"
                    f"{executor_reply.strip()}\n"
                    "</执行者自述>\n\n"
                    "请按系统提示词里的唯一标准裁决，并只调用 release 工具一次。"
                ),
            }
        )
        messages = [
            {"role": "system", "content": self.verifier_prompt},
            {"role": "user", "content": content},
        ]
        self.verifier_calls += 1
        try:
            reply = self.model.chat(
                messages,
                tools=self.verifier_schemas,
                thinking=self.verifier_thinking,
                max_new_tokens=VERIFIER_MAX_NEW_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"核验者调用失败（{type(exc).__name__}: {exc}），按最保守处理"
        parsed = parse_tool_calls(reply)
        for call in parsed:
            if call.name == "release":
                verdict = str((call.args or {}).get("verdict", "")).strip().lower()
                reason = str((call.args or {}).get("reason", "")).strip()
                if verdict == "approve":
                    return True, reason
                if verdict == "reject":
                    return False, reason or "核验者未给出理由"
                return False, f"核验者给出了无法识别的裁决 {verdict!r}：{reason}"
        return False, f"核验者没有给出 release 调用（原文：{reply.strip()[:120]}）"

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
            # The verifier role is told the goal in words, not by replaying the
            # episode: its whole point is a context that does NOT inherit the
            # executor's narrative.
            self._task_text = result.language or result.task
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

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": get_system_prompt(self.prompt_mode) + self.shared_skill + self._skills_index()},
                {
                    "role": "user",
                    "content": [
                        *(
                            [{"type": "image", "image": images[cam]} for cam in ("agent", "wrist") if images.get(cam) is not None]
                            or [{"type": "image", "image": images["agent"]}]
                        ),
                        {
                            "type": "text",
                            "text": (
                                f"任务：{result.language}\n"
                                f"[上图：全局相机 + 腕部相机（从上往下看夹爪下方，用来判断自己在哪）]\n"
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

                call_errors = validate_calls(calls, legacy=self.legacy_tools)
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

                # ---- release gate (two-role: executor context -> verifier) ----
                # Runs BEFORE the tool is dispatched: a rejected release must not
                # touch the arm. The executor's own context is untouched while the
                # verifier runs, so it resumes exactly where it left off.
                _pending = canonical_name(str(call.name))
                if (
                    self.verifier_enabled
                    and _pending == "gripper"
                    and str((call.args or {}).get("action", "")).strip().lower() == "open"
                    and self._grasped
                ):
                    t_v0 = time.time()
                    approved, reason = self._release_verdict(reply, last_images)
                    self._say(
                        f"[turn {turn}] gate: {'放行' if approved else '拒绝'} "
                        f"({time.time() - t_v0:.1f}s) {reason[:80]}"
                    )
                    if not approved:
                        note = (
                            "[释放被否决] 核验者没有放行这次 gripper open，动作**未执行**。"
                            f"理由：{reason}\n"
                            "物体还在夹爪上。请先按上面的理由修正位置，"
                            "确认容器开口在夹爪正下方后再请求放行。"
                        )
                        messages.append({"role": "user", "content": note})
                        transcript.append({"turn": turn, "gate_reject": reason})
                        if recorder is not None:
                            recorder.record(
                                turn,
                                reply=reply,
                                tool_call=f"{_fmt_call(call)} → 被核验者否决",
                                result=note,
                                status="释放被否决",
                                status_color=RED,
                                images=last_images,
                            )
                        continue

                try:
                    t_tool0 = time.time()
                    outcome = self._dispatch(call, obs)
                    outcome["result"] = self._structured_outcome(call, outcome)
                    tool_s = time.time() - t_tool0
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
                        status="任务成功" if outcome.get("done") and outcome.get("success") else "运行中",
                        status_color=GREEN if outcome.get("done") and outcome.get("success") else ACCENT,
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
                    result_text = outcome["text"] + "\n" + json.dumps(
                        outcome["result"], ensure_ascii=False, separators=(",", ":")
                    )
                    messages.append({"role": "user", "content": result_text})
                    transcript.append({"turn": turn, "result": result_text, "success": result.success})
                    if result.success:
                        break
                    continue

                # Track whether the pads actually closed on something. Only
                # `close` reports this, and only the adapter knows the width.
                if _tool == "gripper":
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
                if self.attach_images:
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
                    for cam in ("agent", "wrist"):
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
                        + json.dumps(outcome["result"], ensure_ascii=False, separators=(",", ":"))
                        + "\n" + format_state(obs, self._grasped)
                    ),
                })
                messages.append({"role": "user", "content": content})
                transcript.append({
                    "turn": turn,
                    "result": outcome["text"],
                    "structured_result": outcome["result"],
                })

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
                action = {
                    "kind": "move",
                    "direction": args["direction"],
                    "distance_cm": as_float(args.get("distance_cm")),
                }
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


def _z_err(note: str) -> float | None:
    """Pull 'z err 65.9 mm' out of an adapter note and return it in metres."""
    m = re.search(r"z err ([-0-9.]+) mm", note or "")
    return float(m.group(1)) / 1000.0 if m else None


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
    eef = obs.get("eef_pos", [0, 0, 0])
    width = max(abs(v) for v in obs.get("gripper_qpos", [0, 0]))
    # Same thresholds as `format_outcome`, deliberately. The old rule here was
    # `width > 0.01`, so a width of 0.0230 -- an object between the pads -- was
    # called 张开 in this line while `format_outcome` called the SAME turn
    # "指间有物体（夹住了）". Two vocabularies for one measurement inside a
    # single message is exactly the P0f failure mode.
    if width >= GRIPPER_OPEN_WIDTH:
        grip = "张开"
    elif width <= GRIPPER_EMPTY_WIDTH:
        grip = "闭合（空）"
    else:
        grip = "半闭（指间有物体）"
    # `grasped` comes from the last `set_gripper close` verdict. Do NOT infer it
    # from distance: P0a showed the pads parked 5.5 cm above an untouched can
    # being reported as holding it (7 "slip" + 5 "dropped" false alarms in one
    # run). A grasp is a fact about the gripper.
    held = _carried_name(obs) if grasped else None
    if held:
        # Report HOW FAR the payload sits from the gripper centre. MEASURED
        # (v12): it crept 2.8 -> 3.1 -> 4.3 cm across three `+y 15` carries and
        # fell off on the fourth. The distance grows because the basket is 48 cm
        # from the pick point while one `move` is capped at 15 cm, so the carry
        # is necessarily 4 separate pushes and each one extrudes the can a
        # little further. A number that only ever grows is the only warning the
        # model gets that the grip is failing.
        pos = (obs.get("objects") or {}).get(held)
        if pos is not None:
            gap = float(
                np.linalg.norm(np.asarray(pos, dtype=float) - np.asarray(eef, dtype=float))
            )
            hold = f"爪上：{held}（离爪心 {gap * 100:.1f}cm）"
            # P0b: report the offset (a measurement) and drop both the 3.8cm
            # threshold and the "don't translate" instruction.
            return f"机械爪位置 {_fmt3(eef)} m；夹爪：{grip}；{hold}"
        hold = f"爪上：{held}"
    else:
        hold = "爪上：空（没有物体跟着手）"
    return f"机械爪位置 {_fmt3(eef)} m；夹爪：{grip}；{hold}"


def format_outcome(outcome: dict[str, Any]) -> str:
    kind = outcome.get("kind")
    requested = outcome.get("requested", {})
    delta = outcome.get("measured_delta_m", [0, 0, 0])
    end = outcome.get("eef_end", [0, 0, 0])
    ticks = outcome.get("steps", 0)
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
        text = (
            f"[move {requested.get('direction')} {requested.get('distance_cm')}cm] "
            f"实测位移 {_fmt3(delta)} m（{ticks} ticks）；机械爪现在位于 {_fmt3(end)}"
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
