"""Two transports for the sim server (runs in the `libero` env, login node).

**TCP is the production transport** (`--tcp-port`). It was originally ruled out
by a measurement against 172.18.34.26 -- a different subnet gpu026 cannot
route to. login01 also carries **192.168.82.239/23**, the SAME /23 as gpu026
(192.168.82.46), and that path answers in **0.67 ms** (re-measured 2026-10-08).

The NFS file-drop path is kept for the in-process probes only. It costs **30 s
per call** in the real two-node setup: NFS `acdirmin` defaults to 30 s, so a
command the client publishes stays invisible to the server's directory glob for
up to 30 s. Measured breakdown of one `move`: server-side work 0.56 s,
client-side wait 30.0 s.

Protocol (TCP; 4-byte big-endian length prefix + UTF-8 JSON, frames as base64):
    cmd -> server    {"cmd": "act", "action": {...}}
    server -> cmd    {"id": N, "ok": true, "result": {..., "images_b64": {...}}}

NFS protocol (all paths under <root>/runtime/ipc), for reference:
    client -> server   cmd_<id>.json        published via atomic rename
    server -> client   resp_<id>.json       published via atomic rename
                       img_<id>_<cam>.png   camera frames, if the command made any
"""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import struct
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from arm_agent import contracts as C
from arm_agent.sim.action_adapter import (
    SHORT_BLOCK_ITERS,
    SHORT_BLOCK_MAX_ITERS,
    ActionAdapter,
    ActionOutcome,
    classify_descent,
)
from arm_agent.sim.libero_env import LiberoSim
from arm_agent.sim.pixel_grounding import (
    AIM_MARK_RADIUS_PX,
    Camera,
    Grounding,
    annotate_frame,
    crop_zoom,
    ground_pixel,
)

ZOOM_KEY = C.ZOOM_KEY


class SimServer:
    def __init__(
        self,
        ipc_dir: Path,
        suite_name: str = "libero_object",
        task_id: int = 0,
        cam_size: int = 256,
        seed: int = 0,
        observation_mode: str = "truth",
        pixel_grid_px: int = 32,
        aim_mark_px: int = AIM_MARK_RADIUS_PX,
    ) -> None:
        self.ipc_dir = Path(ipc_dir)
        self.ipc_dir.mkdir(parents=True, exist_ok=True)
        self.sim = LiberoSim(suite_name=suite_name, task_id=task_id, cam_size=cam_size, seed=seed)
        self.adapter = ActionAdapter(self.sim)
        self._images_saved = 0
        self._closing = False
        # 档 B / raw: which channel answers "where is the object".
        #   truth  = the simulator's own body positions (the A/B baseline)
        #   pixels = NO object coordinates anywhere: the model names a pixel and the
        #            code grounds it with Camera + depth.
        #   raw    = the same "no coordinates" rule as pixels, but WITHOUT the pixel
        #            tools: the tool surface is move + gripper + verify, and the model
        #            has to turn what it sees into world-axis motion itself. The camera
        #            returns the plain frame plus the crosshair (its own FK), no grid --
        #            a grid exists to make pixels NAMEABLE, and nothing here names one.
        # The mode is enforced here, at the single place that owns the truth, so a
        # "raw" run cannot silently fall back to names.
        self.observation_mode = str(observation_mode).strip().lower()
        if self.observation_mode not in ("truth", "pixels", "raw"):
            raise ValueError(f"unknown observation_mode {observation_mode!r}")
        self.pixel_grid_px = int(pixel_grid_px)
        self.cam_size = int(cam_size)
        # Radius of the `aim` ring, in pixels of the frame the model receives.
        self.aim_mark_px = int(aim_mark_px)

    # ------------------------------------------------------------------ main
    def serve_forever(self, poll_s: float = C.IPC_POLL_INTERVAL_S, idle_exit_s: float = 3600.0) -> None:
        print(f"[server] listening on {self.ipc_dir} (poll {poll_s * 1000:.0f} ms)", flush=True)
        last_activity = time.time()
        while True:
            handled = self._handle_pending()
            if handled:
                last_activity = time.time()
            else:
                if time.time() - last_activity > idle_exit_s:
                    print("[server] idle timeout, exiting", flush=True)
                    break
                time.sleep(poll_s)
            if self._closing:
                break
        self.sim.close()
        print("[server] stopped", flush=True)

    # -------------------------------------------------------------- commands
    def _handle_pending(self) -> bool:
        handled = False
        for cmd_path in sorted(self.ipc_dir.glob(C.IPC_CMD_GLOB)):
            try:
                cmd_id = int(cmd_path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                cmd_path.unlink(missing_ok=True)
                continue
            try:
                cmd = json.loads(cmd_path.read_text())
                t0 = time.time()
                result = self._dispatch(cmd, cmd_id)
                dt = time.time() - t0
                resp: dict[str, Any] = {"id": cmd_id, "ok": True, "result": result}
                print(f"[server] cmd {cmd_id} {cmd.get('cmd')} ok in {dt:.2f}s", flush=True)
            except Exception as exc:  # noqa: BLE001 - a bad command must not kill the server
                resp = {
                    "id": cmd_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"[server] cmd {cmd_id} FAILED: {exc}", flush=True)
            self._write_atomic(self.ipc_dir / f"resp_{cmd_id}.json", json.dumps(resp))
            cmd_path.unlink(missing_ok=True)
            handled = True
        return handled

    def _dispatch(
        self, cmd: dict[str, Any], cmd_id: int, inline_images: bool = False
    ) -> dict[str, Any]:
        kind = cmd.get("cmd")
        if kind == "ping":
            return {"pong": True, "task": self.sim.task.name}
        if kind == "reset":
            obs = self.sim.reset(init_state_idx=int(cmd.get("init_state_idx", 0)))
            return {"obs": obs, **self._image_payload(cmd_id, inline_images)}
        if kind == "act":
            action = cmd.get("action", {})
            act_kind = action.get("kind")
            # Magnifier request for this command: `[camera_key, u, v]` or None. For
            # `aim` it centres on the ring (the model is judging the ring vs the
            # object); for `target` it centres on the GRIPPER crosshair (the model is
            # judging where the hand ended up vs the object).
            zoom: list[Any] | None = None
            if act_kind == "move":
                outcome = self.adapter.move(
                    str(action["direction"]), float(action["distance_cm"])
                )
                self._attach_pixel_scale(outcome)
            elif act_kind == "rotate":
                outcome = self.adapter.rotate(str(action["axis"]), float(action["angle_deg"]))
                self._attach_pixel_scale(outcome)
            elif act_kind == "gripper":
                outcome = self.adapter.set_gripper(str(action["action"]))
            elif act_kind == "grasp":
                # Concept primitive (档 A): the adapter owns open -> close to
                # contact -> squeeze -> hold window and answers with a status.
                outcome = self.adapter.grasp()
            elif act_kind == "release":
                outcome = self.adapter.release()
            elif act_kind == "align":
                # Server-side primitive: servo the EEF x/y onto an object.
                # See ActionAdapter.align_xy for why the LLM cannot do this.
                name = str(action["name"])
                if self.observation_mode != "truth":
                    raise ValueError(
                        f"observation_mode={self.observation_mode}：不能按物体名对准（那需要仿真真值）。"
                        + self._localisation_hint()
                    )
                # NOTE (fixed): "objects" is SYNTHESISED by pack() from the raw
                # `<name>_pos` keys; it does NOT exist in sim._obs. Reading it
                # off _obs returned {} every time, so align/descend always
                # raised "no object matching ...; have []" (measured: a whole
                # 20-turn episode wasted on that error).
                objects = self.sim.pack(self.sim._obs)["objects"] if self.sim._obs else {}
                key = self._match_object(name, objects)
                if key is None:
                    raise ValueError(f"no object matching {name!r}; have {sorted(objects)}")
                pos = objects[key]
                z_safe = float(action.get("z_safe", 0.22))
                outcome = self.adapter.align_xy((pos[0], pos[1]), z_safe=z_safe)
            elif act_kind == "descend":
                # Aligned descent onto an object: all three axes corrected each
                # tick (align_xy + separate -z was measured to block at can-top
                # height and close on air).
                name = str(action["name"])
                if self.observation_mode != "truth":
                    raise ValueError(
                        f"observation_mode={self.observation_mode}：不能按物体名下降（那需要仿真真值）。"
                        + self._localisation_hint()
                    )
                # Same fix as the align branch: take objects from pack(), not
                # from the raw _obs (which never has an "objects" key).
                objects = self.sim.pack(self.sim._obs)["objects"] if self.sim._obs else {}
                key = self._match_object(name, objects)
                if key is None:
                    raise ValueError(f"no object matching {name!r}; have {sorted(objects)}")
                pos = objects[key]
                z_offset_cm = float(action.get("z_offset_cm", 1.2))
                z_target = pos[2] + z_offset_cm / 100.0
                outcome = self.adapter.descend_to((pos[0], pos[1]), z_target)
            elif act_kind in ("target", "goto_pixel"):
                # 档 B: the model names a PIXEL; the code grounds it with the
                # camera + depth and servos there. No object name, no object
                # position, no object size enters this path. 档 B v2 turns this
                # into a SHORT BLOCK (`max_steps`); `goto_pixel` stays as the
                # pre-v2 name of the same call so old transcripts still replay.
                outcome = self.pixel_target(action, kind=str(act_kind))
                eef = self.sim._obs.get(C.OBS_EEF_POS) if self.sim._obs else None
                if eef is not None:
                    u_eef, v_eef, _ = self._camera("agent").project(np.asarray(eef, dtype=float))
                    if np.isfinite(u_eef) and np.isfinite(v_eef):
                        zoom = ["agent", float(u_eef), float(v_eef)]
            elif act_kind == "aim":
                # 档 B v2 dry run: ground the pixel, mark it, and DO NOT MOVE. The
                # frame that comes back is the same frame `target` would use, so
                # deciding from it is deciding from evidence, not from a guess.
                outcome = self.aim(action)
                zoom = outcome.measurement.get("mark")
            else:
                raise ValueError(f"unknown action kind: {act_kind!r}")
            obs = self.sim.pack(self.sim._obs)  # type: ignore[arg-type]
            mark = outcome.measurement.get("mark")
            return {
                "outcome": outcome.to_dict(),
                "obs": obs,
                **self._image_payload(cmd_id, inline_images, mark=mark, zoom=zoom),
            }
        if kind == "look":
            return {
                "obs": self.sim.pack(self.sim._obs),  # type: ignore[arg-type]
                **self._image_payload(cmd_id, inline_images),
            }
        if kind == "state":
            return {"obs": self.sim.pack(self.sim._obs)}  # type: ignore[arg-type]
        if kind == "success":
            return {"success": self.sim.check_success()}
        if kind == "close":
            self._closing = True
            return {"closing": True}
        raise ValueError(f"unknown command: {kind!r}")

    # ------------------------------------------------------------------ 档 B
    def _attach_pixel_scale(self, outcome) -> None:  # noqa: ANN001 - ActionOutcome
        """Report the step the model just took IN PIXELS, and what a pixel is worth.

        WHY (档 D, three arms measured): the model's own estimate of how far it still has
        to go is **2.4-6x too small** -- median ratio of its claimed distance to the truth
        was 0.42 (`raw` baseline), 0.16 (ladder v1), 0.38 (ladder v2), with 54-72% of the
        quoted moves underestimating by 2x or more. It quotes the pixel gap correctly and
        then converts at roughly **1mm/px, while the global view measures 3.0-4.0mm/px**.
        That single wrong scale makes every step too small in every arm, which is why all
        three crawl and never arrive -- no tool surface fixes it.

        The scale is MEASUREMENT, and the code knows it exactly, so the fix is to hand the
        model the calibration it just performed: this motion moved the crosshair by N
        pixels at this depth. E.g. "一步走了 10cm = 27 像素" is a fact it can read off its
        own action, instead of a scale factor it has to trust.
        """
        start = getattr(outcome, "eef_start", None)
        end = getattr(outcome, "eef_end", None)
        if not (start and end and len(start) == 3 and len(end) == 3):
            return
        try:
            camera = self._camera("agent")
            u0, v0, d0 = camera.project(np.asarray(start, dtype=float))
            u1, v1, d1 = camera.project(np.asarray(end, dtype=float))
        except Exception:  # noqa: BLE001 - a missing camera must not break the motion
            return
        if not all(np.isfinite(x) for x in (u0, v0, u1, v1)):
            return
        outcome.measurement["delta_px"] = [round(float(u1 - u0), 1), round(float(v1 - v0), 1)]
        outcome.measurement["mm_per_px"] = round(float(camera.mm_per_px(float(d1))), 3)

    def _camera(self, key: str) -> Camera:
        cam_name = C.CAMERA_NAMES.get(str(key).strip().lower())
        if cam_name is None:
            raise ValueError(f"unknown camera {key!r}; use one of {sorted(C.CAMERA_NAMES)}")
        # Built fresh on every call: the wrist camera's world pose changes with
        # the arm, and `data.cam_xpos` is what carries that (see Camera.from_env).
        return Camera.from_env(self.sim.env.env, cam_name, self.cam_size, self.cam_size)

    def _log_aim(
        self,
        tool: str,
        cam_key: str,
        camera: Camera,
        pixel: tuple[int, int],
        grounding: Grounding | None,
        reason: str = "",
    ) -> None:
        """The aim instrument: how far from the object did the model actually point?

        LOG ONLY, and it reads ground truth on purpose -- which is why it must never
        reach a result the model sees. It answers the question the two real pixel
        runs left open: "the model cannot read the grid" and "the model reads the grid
        fine but the object is 16 px wide" look identical without it.
        """
        if self.sim._obs is None:
            return
        if grounding is None:
            print(
                f"[aim] tool={tool} asked={pixel} cam={cam_key} REJECTED: {reason}",
                flush=True,
            )
            return
        objects = self.sim.pack(self.sim._obs).get("objects") or {}
        target = np.asarray(grounding.target_xyz, dtype=float)
        rows = []
        for name, pos in objects.items():
            gt = np.asarray(pos, dtype=float)
            gu, gv, _ = camera.project(gt)
            dx, dy = target[0] - gt[0], target[1] - gt[1]
            rows.append(
                (float(np.hypot(dx, dy)), name, (round(gu), round(gv)),
                 (target[2] - gt[2]) * 1000.0)
            )
        rows.sort()
        # ALL objects, not just the two nearest. MEASURED reason: with `rows[:2]` the
        # audit could not say whether the model was aiming at the task's object at all
        # -- a call near three objects logged two of them and the third (the one the
        # run actually cares about) was simply absent, which reads as "no data" instead
        # of "aimed 15 mm from the WRONG object". Six short entries per line is cheap;
        # the alternative is a 50-minute run whose metric cannot be computed.
        near = " | ".join(
            f"{name} px={px} xy={xy * 1000:.0f}mm dz={dz:+.0f}mm"
            for xy, name, px, dz in rows
        )
        print(
            f"[aim] tool={tool} asked={pixel} cam={cam_key} "
            f"target={target.round(3)} depth={grounding.depth_m:.3f}m :: {near}",
            flush=True,
        )

    def _pixel_probe(
        self, action: dict[str, Any]
    ) -> tuple[str, Camera, float, float, Grounding | None, str]:
        """Shared front half of `aim` and `target`: camera, pixel, grounding.

        Returns the camera KEY (`agent`/`wrist`, the name the model used and is shown)
        separately from the Camera, whose `.name` is the physical robosuite camera
        (`agentview` / `robot0_eye_in_hand`). Both appear in the transcript -- the key
        in `requested`, the physical name inside the grounding record -- but only the
        key is ever shown to the model, so "which camera did I ask about" cannot come
        back as a different word than the one it passed.

        An unusable pixel comes back as a REASON, not as an exception: both tools
        want the frame to travel back with the reason so the model can look at it and
        pick another point in the same turn (an exception costs a protocol retry and
        arrives without pixels). Nothing is ever substituted for the point the model
        asked for -- Jev's rule, and its own test name says it best:
        `test_invalid_pixels_cannot_be_clipped_into_another_target`.
        """
        key = str(action.get("camera", "agent")).strip().lower()
        camera = self._camera(key)
        u = float(action["u"])
        v = float(action["v"])
        if not camera.inside(u, v):
            return key, camera, u, v, None, (
                f"像素 ({u:.0f},{v:.0f}) 在 {key} 画面之外"
                f"（画面 u 0..{camera.width - 1}, v 0..{camera.height - 1}）"
            )
        grounding, reason = ground_pixel(
            self.sim.env.env, camera, u, v, action.get("offset_mm") or [0.0, 0.0, 0.0]
        )
        return key, camera, u, v, grounding, reason

    def aim(self, action: dict[str, Any]) -> ActionOutcome:
        """Ground a pixel and DRAW it, without moving anything (档 B v2).

        WHY A TOOL THAT DOES NOTHING. The model's absolute pixel reading carries a
        measured systematic bias (30-50 px = 10-13 cm, stable across four calls at
        512 px), so "look once and jump" cannot be corrected by trying harder -- but
        it CAN be corrected by seeing the error. The ring turns "am I pointing at the
        can?" from an absolute estimate into a relative one ("the ring is two cells
        left of the can"), and the arm pays nothing for the attempt: no motion, no
        physics ticks, no risk of knocking the object it is trying to pick up.

        The ring is drawn on the frame of the camera that was asked about, at exactly
        the pixel the model named (see annotate_frame for why not the round trip).
        """
        cam_key, camera, u, v, grounding, reason = self._pixel_probe(action)
        eef = list(self.sim._obs[C.OBS_EEF_POS]) if self.sim._obs else []
        outcome = ActionOutcome(
            kind="aim",
            requested={"camera": cam_key, "pixel": [round(u), round(v)]},
            eef_start=list(eef),
            eef_end=list(eef),
            measured_delta_m=[0.0, 0.0, 0.0],
        )
        self._log_aim("aim", cam_key, camera, (round(u), round(v)), grounding, reason)
        if grounding is None:
            outcome.status = "rejected"
            outcome.note = reason
            outcome.measurement = {
                "camera": cam_key,
                "pixel": [round(u), round(v)],
                "mark": None,
            }
            return outcome
        outcome.status = "marked"
        outcome.note = "红圈已画在你指定的像素上"
        outcome.measurement = grounding.measurement()
        outcome.measurement["camera"] = cam_key
        outcome.measurement["mark"] = [cam_key, round(u), round(v)]
        return outcome

    def pixel_target(self, action: dict[str, Any], kind: str = "target"):
        """Ground one pixel and servo the gripper there (档 B's replacement for
        `approach(<name>)`), as a SHORT ACTION BLOCK (档 B v2).

        The model supplies the target pixel and the CLEARANCE (`offset_mm`, world
        axes) -- measured reason: the back-projected point sits on the visible
        surface, +20..+90 mm in front of an object's centre, so a hard-coded
        clearance would be wrong for every object but one. An unusable pixel
        (outside the frame, or depth out of range) is REJECTED WITH A REASON
        instead of being quietly replaced by a different target.

        `max_steps` bounds how many xy/z corrections this call may spend (Jev's
        `max_motor_steps`, and its measured motivation on our side: 36 and 19 turns
        of blind `move`s in the two real pixel runs). When the block runs out before
        convergence the outcome says `短块走完还没到`, which is a different fact from
        `blocked` -- the model has to re-look in both cases, but only one of them is
        a reason to change strategy.
        """
        cam_key = str(action.get("camera", "agent")).strip().lower()
        if cam_key == "wrist":
            # MEASURED (scripts/probe_wrist_view.py + the first real pixel run): an
            # eye-in-hand camera makes an absolute pixel DEGENERATE. A fixed pixel
            # is a fixed point in the HAND's frame, so its grounded world point
            # travels with the gripper (measured: pixel (48,150) always lands on
            # the table ~20-30 cm from the pads, wherever the arm is), and no
            # motion can ever remove the error. Episode 0 of the 2026-10-12 run
            # issued 16 identical wrist calls and got `arrived` every time.
            # `aim` is allowed on the wrist view (looking does not move the arm,
            # and the wrist view is the 1.1 mm/px one) -- MOVING to a wrist pixel is
            # not, and this is a wrong-tool error, not an unusable point, so it
            # still raises with the fix in the message.
            raise ValueError(
                "target 拒绝 camera=wrist：腕部相机装在夹爪上，固定像素就是“相对手恒定”的一个点，"
                "它随夹爪一起移动，永远追不上。用 camera=agent 给像素（第三人称）；"
                "想看腕部细部就用 aim(camera=wrist) 只标注不移动，"
                "再配 move 沿世界轴微调：腕部画面里 +y=向左、-y=向右、+x=向上、-x=向下。"
            )
        cam_key, camera, u, v, grounding, reason = self._pixel_probe(action)
        max_steps = action.get("max_steps", SHORT_BLOCK_ITERS)
        try:
            max_steps = int(float(max_steps))
        except (TypeError, ValueError):
            max_steps = SHORT_BLOCK_ITERS
        max_steps = min(max(max_steps, 1), SHORT_BLOCK_MAX_ITERS)
        self._log_aim(kind, cam_key, camera, (round(u), round(v)), grounding, reason)
        if grounding is None:
            eef = list(self.sim._obs[C.OBS_EEF_POS]) if self.sim._obs else []
            return ActionOutcome(
                kind=kind,
                requested={"camera": cam_key, "pixel": [round(u), round(v)]},
                status="rejected",
                note=reason,
                eef_start=list(eef),
                eef_end=list(eef),
                measured_delta_m=[0.0, 0.0, 0.0],
                measurement={"camera": cam_key, "pixel": [round(u), round(v)], "mark": None},
            )
        target = grounding.target_xyz
        outcome = self.adapter.descend_to(
            (target[0], target[1]), target[2], max_iters=max_steps
        )
        outcome.kind = kind
        outcome.requested = {
            "camera": cam_key,
            "pixel": [grounding.pixel[0], grounding.pixel[1]],
            "offset_mm": [round(float(x), 1) for x in action.get("offset_mm") or [0.0, 0.0, 0.0]],
            "max_steps": max_steps,
        }
        # MERGE into the grounding record, never replace it: `descend_to` has already
        # written `iterations` / `truncated` into this dict, and overwriting it is the
        # same mistake that once hid `eef_start`/`eef_end` from the audit.
        outcome.measurement = {**grounding.measurement(), **outcome.measurement}
        outcome.measurement["camera"] = cam_key
        for key in ("eef_start", "eef_end", "measured_delta_m", "steps"):
            value = getattr(outcome, key, None)
            if value is not None:
                outcome.measurement[key] = value
        outcome.status = classify_descent(outcome)
        outcome.note = f"{kind} {outcome.status}: {outcome.note}"
        return outcome

    def _localisation_hint(self) -> str:
        """What this mode offers INSTEAD of names (one sentence, per mode).

        The refusal has to teach the alternative, or the model spends a turn guessing
        which surface it is on: `pixels` still has a way to say "that one", `raw` has
        none by design and has to be told so.
        """
        if self.observation_mode == "pixels":
            return "请改用 aim 确认像素、再用 target 移动。"
        return "本模式没有任何定位工具：只能 move + gripper，靠画面自己换算方向与距离。"

    def _annotate(
        self,
        cam_key: str,
        image_display: np.ndarray,
        mark: list[Any] | None = None,
    ) -> np.ndarray:
        """Grid + gripper crosshair (+ the caller's ring), only in `pixels` mode.

        The grid is what makes a pixel NAMEABLE by the model, and the crosshair is
        where the gripper actually is (its own proprioception, projected) -- the
        pair replaces "the tool told me where the object is". The optional `mark`
        is `aim`'s dry run: `[camera_key, u, v]`, drawn on that camera only, so
        "where I asked" and "where my hand is" can be compared in one frame.
        """
        if self.observation_mode == "truth":
            return image_display
        try:
            camera = self._camera(cam_key)
            eef = self.sim._obs.get("robot0_eef_pos") if self.sim._obs else None
            ring = None
            if mark and str(mark[0]) == str(cam_key):
                ring = (float(mark[1]), float(mark[2]))
            return annotate_frame(
                image_display,
                camera,
                eef,
                self.pixel_grid_px,
                mark=ring,
                mark_radius=self.aim_mark_px,
                draw_grid=self.observation_mode != "raw",
            )
        except Exception as exc:  # noqa: BLE001 - annotation must never break a step
            print(f"[server] annotate failed: {type(exc).__name__}: {exc}", flush=True)
            return image_display

    # ------------------------------------------------------------------ util
    @staticmethod
    def _match_object(query: str, objects: dict[str, Any]) -> str | None:
        """Fuzzy object lookup, shared by locate (agent side) and align (here)."""
        query_norm = str(query).strip().lower().replace(" ", "_")
        if query_norm in objects:
            return query_norm
        for key in objects:
            stem = key.rsplit("_", 1)[0].lower()
            if query_norm and (query_norm in key.lower() or query_norm in stem or stem in query_norm):
                return key
        return None

    def _image_payload(
        self,
        cmd_id: int,
        inline: bool,
        mark: list[Any] | None = None,
        zoom: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Camera frames for a response: base64-inline (TCP) or on disk (NFS).

        `mark` and `zoom` are deliberately parameters and not server state: a ring
        left over from an earlier `aim` would read as "your point is here" on a frame
        the model is using for a DIFFERENT decision, i.e. it would be a silently wrong
        target. `zoom` is `[camera_key, u, v]`, the centre of the magnified crop.
        """
        if inline:
            return {"images_b64": self._images_b64(mark, zoom)}
        return {"images": self._save_images(cmd_id, mark, zoom)}

    def _images_b64(
        self, mark: list[Any] | None = None, zoom: list[Any] | None = None
    ) -> dict[str, str]:
        """Both camera frames as base64 PNGs (display orientation) + the magnifier."""
        from PIL import Image

        self.sim.refresh_images()
        out: dict[str, str] = {}
        for cam, frame in self.sim.images().items():
            buf = io.BytesIO()
            display = np.asarray(frame)[::-1]
            annotated = self._annotate(cam, display, mark)
            Image.fromarray(annotated).save(buf, format="PNG")
            out[cam] = base64.b64encode(buf.getvalue()).decode("ascii")
            if zoom and str(zoom[0]) == str(cam):
                crop = crop_zoom(annotated, (float(zoom[1]), float(zoom[2])))
                if crop is not None:
                    zbuf = io.BytesIO()
                    Image.fromarray(crop).save(zbuf, format="PNG")
                    out[ZOOM_KEY] = base64.b64encode(zbuf.getvalue()).decode("ascii")
        return out

    def _save_images(
        self,
        cmd_id: int,
        mark: list[Any] | None = None,
        zoom: list[Any] | None = None,
    ) -> dict[str, str]:
        """Persist both camera frames as PNG; returns camera -> relative path."""
        from PIL import Image

        # Idempotent: renders only if the cameras were disabled during internal
        # ticks (or never refreshed); otherwise it is a no-op.
        self.sim.refresh_images()
        paths: dict[str, str] = {}
        for cam, frame in self.sim.images().items():
            name = f"img_{cmd_id}_{cam}.png"
            # Frames are stored vertically flipped for display (LIBERO
            # convention); flip once here so every consumer sees the same
            # display orientation the E6 probe validated. Same annotation helper
            # as the TCP path, so the two transports cannot drift apart.
            display = np.asarray(frame)[::-1]
            annotated = self._annotate(cam, display, mark)
            Image.fromarray(annotated).save(self.ipc_dir / name)
            paths[cam] = name
            if zoom and str(zoom[0]) == str(cam):
                crop = crop_zoom(annotated, (float(zoom[1]), float(zoom[2])))
                if crop is not None:
                    zname = f"img_{cmd_id}_{ZOOM_KEY}.png"
                    Image.fromarray(crop).save(self.ipc_dir / zname)
                    paths[ZOOM_KEY] = zname
        self._images_saved += 1
        return paths

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.rename(tmp, path)  # NFS rename is atomic (measured)

    # ------------------------------------------------------------------ tcp
    def serve_tcp(self, port: int, idle_exit_s: float = 3600.0) -> None:
        """Serve the protocol over TCP (production path).

        Accepts clients one at a time and processes commands strictly in order,
        mirroring the file-drop semantics. Frames travel base64-inline, so
        neither side ever waits on NFS attribute-cache expiry (that cost 30 s
        per call, measured). Accepting in a LOOP lets a short-lived liveness
        probe (`cli.py ipc-ping`) connect and disconnect without ending the
        session.
        """
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(4)
        print(f"[server] TCP listening on 0.0.0.0:{port}", flush=True)
        cmd_id = 0
        try:
            while not self._closing:
                srv.settimeout(max(1.0, idle_exit_s))
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    print("[server] no client arrived, idle exit", flush=True)
                    break
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[server] client connected from {addr}", flush=True)
                try:
                    while not self._closing:
                        conn.settimeout(max(1.0, idle_exit_s))
                        header = conn.recv(4)
                        if not header:
                            print("[server] client disconnected", flush=True)
                            break
                        while len(header) < 4:
                            more = conn.recv(4 - len(header))
                            if not more:
                                raise ConnectionResetError("header truncated")
                            header += more
                        size = struct.unpack(">I", header)[0]
                        body = b""
                        while len(body) < size:
                            chunk = conn.recv(size - len(body))
                            if not chunk:
                                raise ConnectionResetError("body truncated")
                            body += chunk
                        cmd = json.loads(body)
                        cmd_id += 1
                        t0 = time.time()
                        try:
                            result = self._dispatch(cmd, cmd_id, inline_images=True)
                            resp: dict[str, Any] = {"id": cmd_id, "ok": True, "result": result}
                            print(
                                f"[server] cmd {cmd_id} {cmd.get('cmd')} ok in "
                                f"{time.time() - t0:.3f}s",
                                flush=True,
                            )
                        except Exception as exc:  # noqa: BLE001 - a bad command must not kill the server
                            resp = {
                                "id": cmd_id,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                            }
                            print(f"[server] cmd {cmd_id} FAILED: {exc}", flush=True)
                        out = json.dumps(resp).encode("utf-8")
                        conn.sendall(struct.pack(">I", len(out)) + out)
                except (ConnectionResetError, socket.timeout, OSError) as exc:
                    print(f"[server] connection ended: {type(exc).__name__}", flush=True)
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
        finally:
            srv.close()
            self.sim.close()
            print("[server] stopped", flush=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="arm_agent sim server (login node)")
    parser.add_argument("--ipc-dir", required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--cam", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idle-exit", type=float, default=3600.0)
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=None,
        help="serve over TCP on this port (production path); omit to use the NFS file-drop",
    )
    args = parser.parse_args(argv)

    server = SimServer(
        ipc_dir=Path(args.ipc_dir),
        suite_name=args.suite,
        task_id=args.task_id,
        cam_size=args.cam,
        seed=args.seed,
    )
    if args.tcp_port:
        server.serve_tcp(port=args.tcp_port, idle_exit_s=args.idle_exit)
    else:
        server.serve_forever(idle_exit_s=args.idle_exit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
