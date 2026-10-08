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
from arm_agent.agent.parser import ToolCall, as_float, parse_tool_calls, truncate_to_first_call
from arm_agent.agent.prompts import TOOL_SCHEMAS, get_system_prompt
from arm_agent.agent.render import ACCENT, EpisodeRecorder, GREEN, RED, YELLOW

SKILLS_DIRNAME = "skills"
# Feed-back images: only the most recent N frames stay as pixels; the rest are
# text stubs. Was 2 until v7 -- the prompt grew, and one extra frame's image
# tokens plus the longer system prompt pushed the unfused attention over the
# top. 1 frame is the newest view, which is the one the decision needs.
HISTORY_IMAGE_KEEP = 1
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
MAX_HISTORY_MESSAGES = 4


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
    ) -> None:
        self.client = client
        self.model = model
        self.workspace = Path(workspace)
        self.max_turns = max_turns
        self.attach_images = attach_images
        self.log = log
        # "full" = the accumulated scene-specific prompt; "plain" = task +
        # frame + tool semantics only (see prompts.get_system_prompt).
        self.prompt_mode = prompt_mode
        self.skills_dir = self.workspace / SKILLS_DIRNAME
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        # Last observed grip offset, used to detect the payload creeping out of
        # the pads (see the slip-trend block in run_episode).
        self._prev_gap: float | None = None

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

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": get_system_prompt(self.prompt_mode) + self._skills_index()},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images["agent"]},
                        {
                            "type": "text",
                            "text": (
                                f"任务：{result.language}\n"
                                f"[上图是全局相机看到的场景]\n"
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
                        prepare_for_model(compact_history(messages), HISTORY_IMAGE_KEEP),
                        tools=TOOL_SCHEMAS,
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

                if len(calls) != 1:
                    result.protocol_retries += 1
                    note = (
                        "你的上一条回复没有包含工具调用或包含多个。"
                        "请只输出**一个** <tool_call>...</tool_call> 块。"
                    )
                    messages.append({"role": "user", "content": note})
                    transcript.append({"turn": turn, "protocol_retry": True})
                    self._say(f"[turn {turn}] protocol retry (calls={len(calls)})")
                    if recorder is not None:
                        recorder.record(
                            turn,
                            reply=reply,
                            tool_call=f"(协议失败: 解析出 {len(calls)} 个调用)",
                            result=note,
                            status="协议重试",
                            status_color=YELLOW,
                            images=last_images,
                        )
                    continue

                call = calls[0]
                result.tool_calls += 1
                self._say(f"[turn {turn}] {call.name} {call.args}")

                try:
                    t_tool0 = time.time()
                    outcome = self._dispatch(call, obs)
                    tool_s = time.time() - t_tool0
                except Exception as exc:  # noqa: BLE001 - tool errors are prompts, not crashes
                    text = f"[工具错误] {type(exc).__name__}: {exc}。请修正参数后重试。"
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
                    messages.append({"role": "user", "content": outcome["text"]})
                    transcript.append({"turn": turn, "result": outcome["text"], "success": result.success})
                    if result.success:
                        break
                    continue

                content: list[dict[str, Any]] = []
                if outcome.get("image") is not None and self.attach_images:
                    content.append({"type": "image", "image": outcome["image"]})

                # Grip-slip TREND, not just the current value.
                #
                # MEASURED (v12/v13): the can crept 2.8 -> 3.1 -> 4.3 cm off the
                # gripper centre across three `+y 15` carries and fell off on the
                # fourth (the basket is 48 cm from the pick point and one `move`
                # is capped at 15 cm, so the carry is necessarily 4 pushes).
                # A static threshold cannot catch this: 3.1 cm looks harmless.
                # And the model cannot see the trend by itself -- history is
                # capped at MAX_HISTORY_MESSAGES, so it only ever sees the most
                # recent number. Compute the delta here and say it out loud.
                gap = _hold_gap(obs)
                slip = ""
                if gap is None:
                    if self._prev_gap is not None:
                        # THE event that matters. MEASURED (v14 t16): the can was
                        # in the pads on t13 (gap 3.1 cm) and simply gone on t16.
                        # The distance-trend check below can NEVER fire for this:
                        # the moment the payload leaves the pads it also leaves
                        # `_carried_name`'s 6 cm radius, so `gap` jumps to None
                        # and there is no sample between "3.1 cm" and "gone".
                        # v12 only looked detectable because that run happened to
                        # dawdle at 4.3 cm for a turn.
                        # Detect the DISAPPEARANCE, not just the drift.
                        slip = (
                            "\n⛔ **物体掉了**（上一轮还在指间）。先 `locate` 看它掉在哪："
                            "够得到就重新抓；若报“够不到”，用爪子在它侧面轻推（move ±x 5~10cm）挪到够得到的区域。"
                        )
                    self._prev_gap = None  # nothing held: reset the trend
                else:
                    if self._prev_gap is not None and gap > self._prev_gap + 0.003:
                        slip = (
                            f"\n⚠ **正在往外滑**（离爪心 {self._prev_gap * 100:.1f}→{gap * 100:.1f}cm）。"
                            "**别再横移**：到篮子附近就立刻投放，否则先放回桌面重抓。"
                            "（close 无效——指垫已闭到底）"
                        )
                    elif gap > 0.038:
                        slip = (
                            f"\n⚠ **快滑脱了**（离爪心 {gap * 100:.1f}cm）。**别再横移**：到篮子附近立刻投放，否则放回桌面重抓。"
                        )
                    self._prev_gap = gap
                content.append({
                    "type": "text",
                    "text": outcome["text"] + "\n" + format_state(obs) + slip,
                })
                messages.append({"role": "user", "content": content})
                transcript.append({"turn": turn, "result": outcome["text"] + slip})

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
            if z_err is not None and z_err > 0.02:
                return {
                    "obs": resp.obs,
                    "text": (
                        f"[descend_to {object_name} +{z_offset}cm] ⚠ **够不到**：xy 已对准，"
                        f"但爪子只降到 z={end[2]:.4f}，离目标还差 {z_err * 100:.1f}cm"
                        "（这是工作空间限制，不是没对准）。\n"
                        "  在同一个位置**再试 descend_to 也不会更低**。可选做法：\n"
                        "  ① 用爪子在物体侧面轻推，把它挪到别处（例如 move ±x 5~10cm），再 locate 重新对准；\n"
                        "  ② 从物体的另一侧靠近再试。\n"
                        "  **不要**直接 set_gripper close —— 够不到，一定夹空。"
                    ),
                    "image": images.get("agent"),
                    "images": images,
                    "done": False,
                }
            return {
                "obs": resp.obs,
                "text": (
                    f"[descend_to {object_name} +{z_offset}cm] 已对准并下降到 "
                    f"({end[0]:.4f}, {end[1]:.4f}, {end[2]:.4f})（{note}，"
                    f"{outcome.get('steps', 0)} ticks）。下一步用 set_gripper 操作夹爪。"
                ),
                "image": images.get("agent"),
                "images": images,
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
                    "下一步可以 move -z 下降到抓取高度（目标 z 见 locate 的提示）。"
                ),
                "image": images.get("agent"),
                "images": images,
                "done": False,
            }

        if name == "locate":
            return {"obs": obs, "text": self._locate(args.get("name", ""), obs), "image": None, "done": False}

        if name == "get_state":
            return {"obs": obs, "text": f"[get_state] {format_state(obs)}", "image": None, "done": False}

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
            }

        if name == "declare_done":
            success = self.client.success()
            if success:
                text = f"[declare_done] 验证通过，任务完成！说明：{args.get('answer_note', '')}"
            else:
                text = (
                    "[declare_done] 任务完成检查**未通过**：目标物体还没有按要求放好。"
                    "请继续观察和调整（可以先 look 或 get_state 确认现状）。"
                )
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

        def hint(axis_letter: str, value: float) -> str:
            if abs(value) < 0.005:
                return f"{axis_letter} 轴已对齐"
            direction = ("+" if value > 0 else "-") + axis_letter
            cm = abs(value) * 100
            # Single-call limit, kept in sync with adapter.MOVE_MAX_DISTANCE_CM.
            # One long move is safer than several short ones (each move has a
            # small chance of shaking the payload loose).
            if cm > 15.0:
                repeats = int(cm // 15) + 1
                return f"需要 move {direction} 15.0cm 约 {repeats} 次（共 {cm:.0f}cm）"
            return f"move {direction} {cm:.1f}cm"

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
        holding = _carried_name(obs) == match and hold_gap < 0.06

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
                f"  ⚠ **它正被你的夹爪抓着**（离爪心 {hold_gap * 100:.1f}cm）——上面三个数字只是说明它贴着你，"
                "**不要**按它们的差值移动，也**不要**再 descend_to / close 去“重新抓”。",
            ]
            if pos[2] < CARRY_SAFE_CENTRE_Z:
                lines.append(
                    f"  ⛔ **高度不够：现在横移会把篮子撞飞。** 它的中心 z={pos[2]:.4f}，"
                    f"底部大约 z={pos[2] - 0.054:.4f}，而**篮口在 z≈{BASKET_RIM_Z:.3f}**——底部比篮口低，"
                    "贴着它扫过去就会被刮掉（这就是前几次掉在篮子附近的原因）。\n"
                    f"    先 `move +z` 把它抬到**中心 z ≥ {CARRY_SAFE_CENTRE_Z:.2f}**"
                    f"（即再抬 {(CARRY_SAFE_CENTRE_Z - pos[2]) * 100:.0f}cm 以上），locate 确认 z 够了再横移。\n"
                    "    **不要超过 z≈0.30**（再高会接近工作空间上限）。"
                )
            else:
                lines.append(
                    f"  ✓ 高度 OK（中心 z={pos[2]:.4f}，底部高于篮口 {BASKET_RIM_Z:.3f}），可以横移。"
                )
            if hold_gap > 0.038:
                lines.append(
                    f"  ⚠ **它在往外滑**（离爪心 {hold_gap * 100:.1f}cm，健康 <3cm）。"
                    "**别再横移**：到篮子附近立刻投放，否则放回桌面重抓。（close 无效）"
                )
            lines.append(
                "  下一步：locate basket，按**篮子那一条**给出的方向移动；"
                "到篮子正上方后 descend_to(basket, 15) → set_gripper open → declare_done。"
            )
            lines.append("  （除非画面显示它其实已经掉了，那时才按掉的流程处理。）")
        else:
            lines = [
                f"[locate] {match}: ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m",
                f"  机械爪当前: ({eef[0]:+.4f}, {eef[1]:+.4f}, {eef[2]:+.4f}) m",
                f"  偏差（物体 - 机械爪）: dx={dx * 100:+.1f}cm, dy={dy * 100:+.1f}cm, dz={dz * 100:+.1f}cm",
                f"  下一步建议: 先 {hint('x', dx)}；再 {hint('y', dy)}（一次只做一个，做完复查）",
                f"  抓取高度: x/y 对齐后降到 z≈{pos[2] + 0.012:.3f} m",
                "  注意: 单次 move 可走 0.1~15cm，尽量一次走完；偏差小于 5cm 时按实际偏差移动，别固定用 5cm（会过冲震荡）。",
            ]
        # Proximity warning. MEASURED (probe_descend_near_basket): descending
        # within 4 cm of the basket shoves it 10-37 mm, while at >=6 cm it is
        # untouched -- i.e. a re-grasp next to the basket reliably knocks it over
        # (that is what killed the v5/v6 runs). Tell the model BEFORE it descends.
        for other_name, other_pos in (obs.get("objects") or {}).items():
            if other_name == match:
                continue
            d_xy = float(np.linalg.norm(np.asarray(other_pos, dtype=float)[:2] - np.asarray(pos, dtype=float)[:2]))
            if d_xy < 0.07:
                if holding:
                    lines.append(
                        f"  ⚠ 注意避让: {other_name} 距它只有 {d_xy * 100:.1f}cm，搬着它横移时别撞到。"
                    )
                else:
                    lines.append(
                        f"  ⚠ 危险邻近: {other_name} 距它只有 {d_xy * 100:.1f}cm，"
                        "4cm 以内下降必把它推开 10~37mm。别在这个距离上下降/横移；"
                        "先把物体推离到 >10cm 再抓。"
                    )
        return "\n".join(lines)

    def _skills_index(self) -> str:
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
# Basket rim height, measured by probe_basket_geometry (the pads cannot descend
# below it). Anything carried lower than this sweeps the rim when translating.
BASKET_RIM_Z = 0.115
# Centre height a carried object needs so that its BOTTOM clears the rim. The
# can is ~10.7 cm tall (half = 0.054), so centre >= 0.115 + 0.054 = 0.169; use a
# round 0.17 and let the message state the reasoning.
CARRY_SAFE_CENTRE_Z = 0.17


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


def _hold_gap(obs: dict[str, Any]) -> float | None:
    """Distance from the gripper centre to the object riding in the pads.

    This is the grip-slip indicator. None when nothing is held.
    """
    import numpy as _np

    held = _carried_name(obs)
    if not held:
        return None
    pos = (obs.get("objects") or {}).get(held)
    if pos is None:
        return None
    eef = obs.get("eef_pos")
    if eef is None:
        return None
    return float(_np.linalg.norm(_np.asarray(pos, dtype=float) - _np.asarray(eef, dtype=float)))


def format_state(obs: dict[str, Any]) -> str:
    eef = obs.get("eef_pos", [0, 0, 0])
    width = max(abs(v) for v in obs.get("gripper_qpos", [0, 0]))
    grip = "张开" if width > 0.01 else "闭合"
    held = _carried_name(obs)
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
            if gap > 0.038:
                # Do NOT advise "close to re-tighten": the pads are already shut
                # (close is idempotent), so that can only squeeze the payload
                # further out. MEASURED (v14 t17): the very turn that ran
                # `set_gripper close` is the turn the can had already left.
                hold += " ⚠ 快滑脱了，别再横移。"
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
            text += (
                f" ⚠ **只走了 {got * 100:.1f}cm（要求 {want * 100:.1f}cm）**"
                "——被挡住，或这个方向已经到极限。不要重复同一个方向；换个方向，"
                "或先抬高再横移绕开。"
            )
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
                verdict = (
                    " ⚠ 夹空了（两指完全闭合）。重试：先 move +z 5 抬起，再 align_xy，再 descend_to(name, 1.2)；"
                    "别用负 z_offset 硬压（会把物体横向推开）。"
                )
            elif width >= GRIPPER_OPEN_WIDTH:
                verdict = " ⚠ 夹爪没有闭合（仍在张开位置）——请确认已对准物体再 close。"
            else:
                verdict = " ✓ 指间有物体（夹住了）。下一步可以提起并运送到篮子。"
        elif action == "open":
            # Tell the model what an open means RIGHT HERE, every time. The v4
            # episode failed because it read `爪上：空` after releasing the can
            # into the basket, concluded "it fell out", and spent 36 turns
            # chasing it. `爪上：空` is EXPECTED after a deliberate release.
            verdict = " ✓ 已张开（物体已脱手）。如果你是把它放进容器，这就是正常的，请接着调用 declare_done 让系统验证，不要再去抓它。"
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
            "如需要，可用 get_state 或 look 重新获取当前状态。"
        ),
    }
    return [*head, note, *tail]
