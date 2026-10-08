"""Semantic action -> OSC delta sequence translation.

This is the project's documented "accident-prone area" (DESIGN.md 2.3), so
every convention below is measured or source-read, never guessed:

- Position deltas are base-frame xyz, applied as `goal = current + delta`
  (robosuite/utils/control_utils.py::set_goal_position).
- Rotation deltas are base-frame axis-angle, applied as `goal = R_err @ R`
  (left multiply, set_goal_orientation).
- Input [-1, 1] scales linearly to +/-[0.05 m, 0.5 rad] (load_controller_config
  measured values), so one env.step() moves at most 5 cm / 28.6 deg.
- Gripper: -1 opens, +1 closes, but current_action slews at 0.01/step, so the
  command must be HELD for ~100 steps; we hold until the fingers converge.
- During translation/rotation the gripper channel sends 0.0 = "hold": sign(0)=0
  leaves current_action unchanged, so the fingers keep their last target.

The adapter also returns the MEASURED outcome of every action (delta between
before/after eef positions) so the harness can feed real numbers back to the
model instead of assuming the motion happened.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from arm_agent import contracts as C

if TYPE_CHECKING:
    from arm_agent.sim.libero_env import LiberoSim

# How much of one step's 5 cm budget to use per internal step. Half budget is
# smoother (each OSC tick aims at a nearby goal, so tracking error stays low).
TRANSLATE_STEP_CM = 2.5
# Rotation per internal step, below the 28.6 deg cap for the same reason.
ROTATE_STEP_DEG = 10.0

# Closed-loop arrival thresholds. MEASURED (probe_action_semantics.py): the OSC
# controller recomputes its goal as `current + delta` every tick, so a single
# open-loop delta lands only ~0.3 cm of the requested 2.5 cm (the arm is still
# tracking when the next target replaces it). Requesting 5 cm in 2 open-loop
# steps therefore moved just 0.66 cm. The adapter instead commands
# `target - current` each tick until the EEF is within tolerance, which is both
# accurate and self-limiting.
#
# Tolerance is 2 mm, NOT 5 mm: a 3 mm request (the dry-run policy asked for
# 0.3 cm at the end of its retreat) would otherwise be swallowed by a 5 mm
# window and the move would execute 0 ticks while reporting success.
MOVE_ARRIVE_TOL_M = 0.002
MOVE_MAX_STEPS = 30  # 1.5 s of control time; measured need is ~4-15 ticks
# Early abort for a BLOCKED axis. When the servo commands motion but the EEF
# barely moves, it is pushing against the can top / table. Burning the full 30
# ticks there is pure waste: measured, descend_to spent 681 of its 984 ticks in
# blocked -z attempts (-z avg 27.2 ticks/call vs 7.5 for a free move).
MOVE_STALL_EPS_M = 0.0005  # < 0.5 mm of travel in one tick = not tracking
MOVE_STALL_PATIENCE = 5
# Per-tick displacement cap for the closed-loop servo.
# NOTE (measured): slowing this to 2.5 cm/tick changed the delicate descent
# dynamics so that the pads closed on air (width 0.0005 vs 0.0194 body contact
# at 5 cm/tick). Keep the controller's full 5 cm budget.
# GENTLE mode -- WHY (measured, probe_move_dynamics + probe_carry_modes +
# probe_grasp_depth):
#   * Carrying ALONG the gripper's opening axis sheds the payload, because its
#     inertia then pushes straight out of the pads instead of being resisted
#     along the pad faces: carry.opening-axis cos = 0.02 -> HELD, 0.62 -> DROPPED.
#   * But even PERPENDICULAR motion slides a SHALLOW grip: with |eef-can| = 26 mm
#     the can crept ~6 mm during the +x leg and the +y leg then flicked it out,
#     while the deep grip (13 mm, 1 cm/tick throughout) held.
# So the condition is "carrying", not "direction": any move with an object in
# the pads runs at this reduced per-tick displacement.
# Not a global slowdown: 2.5 cm/tick for the DESCENT makes the pads close on air
# (width 0.0007), which is why MOVE_STEP_CAP_M keeps the full 5 cm budget and
# this is gated on `_carrying()` (the descent happens with empty pads).
MOVE_STEP_CAP_M = C.MAX_POS_STEP_M
GENTLE_STEP_M = 0.01
GENTLE_AXIS_DOT = 0.5  # |cos| with the opening axis, logged when applicable
# The tool contract is 0.1..15 cm per call and the prompt tells the model the
# same. Measured (probe_move_chunking): the payload is shed on only ~4% of
# moves, but that compounds -- a 60 cm trip is ~12 moves at 5 cm (≈39% chance of
# dropping it somewhere) versus ~4 moves at 15 cm (≈15%). A single 15 cm servo
# tested 6/6 HELD, so the longer step is not riskier per move.
MOVE_MAX_DISTANCE_CM = 15.0
ROTATE_ARRIVE_TOL_DEG = 3.0
ROTATE_MAX_STEPS = 30

# Gripper convergence thresholds (rad). Open qpos ~= 0.0208, closed ~= 0.
GRIPPER_CONVERGE_EPS = 0.002
# MEASURED finger width (robosuite Panda), used to turn a raw qpos into a
# verdict instead of guessing:
#   fully open ........ ~0.0388
#   shut on air ....... ~0.0007
#   object between .... ~0.018-0.030   <-- a SUCCESSFUL grasp lands here
# The old label was `width > 0.01 -> "open"`, which called a real grasp (0.030)
# "open" too.
GRIPPER_OPEN_WIDTH = 0.034
GRIPPER_EMPTY_WIDTH = 0.006
GRIPPER_SETTLE_AVG = 5  # finger must move < eps per step for this many steps
GRIPPER_MAX_STEPS = 130  # measured: full sweep needs ~100 steps at 0.01/step
# Extra full-power ticks after the pads stop moving on an object. The clamp
# command is rate-limited (0.01/tick) so convergence fires with low force, but
# a long ramp (measured: 100 ticks) SQUEEZES THE CAN OUT of the grip
# (close width 0.0005 afterwards) and 30 ticks let it slip mid-carry.
# 10 ticks firms the pinch without ejecting.
GRIPPER_RAMP_EXTRA_STEPS = 10

AXIS_VECTORS = {
    # rotate tool axis name -> base-frame axis-angle unit axis
    "yaw": (0.0, 0.0, 1.0),  # about world z
    "pitch": (0.0, 1.0, 0.0),  # about world y
    "roll": (1.0, 0.0, 0.0),  # about world x
}

DIRECTION_VECTORS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


@dataclass
class ActionOutcome:
    """What actually happened, in numbers the model can reason about."""

    kind: str
    requested: dict[str, Any]
    steps: int = 0
    eef_start: list[float] = field(default_factory=list)
    eef_end: list[float] = field(default_factory=list)
    measured_delta_m: list[float] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "requested": self.requested,
            "steps": self.steps,
            "eef_start": self.eef_start,
            "eef_end": self.eef_end,
            "measured_delta_m": self.measured_delta_m,
            "note": self.note,
        }


class ActionAdapter:
    def __init__(self, sim: "LiberoSim") -> None:
        self.sim = sim

    @contextlib.contextmanager
    def _renderless(self):
        """Run internal ticks without cameras (90% of step time), then refresh."""
        self.sim.set_render(False)
        try:
            yield
        finally:
            self.sim.refresh_images()

    # ------------------------------------------------------------- translate
    def move(self, direction: str, distance_cm: float) -> ActionOutcome:
        vec = np.asarray(DIRECTION_VECTORS[direction], dtype=np.float64)
        axis_index = int(np.argmax(np.abs(vec)))
        # Gentle mode: while the pads hold something, run the servo slowly.
        # A shallow grip slides even on the safe direction, and along the
        # opening axis inertia pushes the payload straight out (see
        # GENTLE_STEP_M). The descent runs with empty pads, so it keeps full
        # speed -- which matters, because slow descent closes on air.
        cap = MOVE_STEP_CAP_M
        gentle = False
        align = 0.0
        if self._carrying():
            opening = self._opening_axis()
            if opening is not None:
                align = abs(float(vec @ opening))
            cap = GENTLE_STEP_M
            gentle = True
        # Clamp to the documented tool range (0.1..5 cm); the harness prompt
        # tells the model the same range.
        distance_cm = float(np.clip(distance_cm, 0.1, MOVE_MAX_DISTANCE_CM))
        # Arrival tolerance must scale with the request. MEASURED (v8 t51/t55/
        # t58/t61): `move ... 0.1cm` (1 mm) is BELOW MOVE_ARRIVE_TOL_M (2 mm), so
        # the loop below broke on its first iteration, ran 0 ticks, and reported
        # a (0,0,0) displacement -- indistinguishable from "blocked". The model
        # read that as the arm being stuck and retried the same tiny move.
        # A requested nudge has to actually be attempted.
        arrive_tol = min(MOVE_ARRIVE_TOL_M, abs(distance_cm) / 100.0 * 0.5)
        start = self._eef_now()
        target_axis = float(start[axis_index]) + float(np.sign(vec[axis_index])) * distance_cm / 100.0
        outcome = ActionOutcome(
            kind="move",
            requested={"direction": direction, "distance_cm": distance_cm},
            eef_start=list(start),
        )
        with self._renderless():
            # STRICTLY single-axis (measured fix). The old version commanded the
            # full 3-D error vector (`action[:3] = (target - eef)/cap`), so any
            # lateral drift the OSC couples into a descent was chased by `move`
            # itself; descend_to then burned 760 of its 1264 ticks re-aligning
            # (off-axis -x/+y corrections dominated). align_xy and _move_axis_z
            # were already single-axis; `move` must honour its own
            # "沿世界坐标轴平移" contract.
            prev_axis = float(start[axis_index])
            stall = 0
            for _ in range(MOVE_MAX_STEPS):
                err = target_axis - float(self._eef_now()[axis_index])
                if abs(err) < arrive_tol:
                    break
                step = max(-MOVE_STEP_CAP_M, min(MOVE_STEP_CAP_M, err))
                action = np.zeros(C.ACTION_DIM)
                action[axis_index] = step / C.MAX_POS_STEP_M
                action[6] = C.GRIPPER_HOLD
                self.sim.step(action)
                outcome.steps += 1
                cur_axis = float(self._eef_now()[axis_index])
                if abs(cur_axis - prev_axis) < MOVE_STALL_EPS_M:
                    stall += 1
                    if stall >= MOVE_STALL_PATIENCE:
                        outcome.note = (
                            f"blocked on {direction}: commanded {step * 100:+.1f}cm/"
                            f"tick but moved <{MOVE_STALL_EPS_M * 1000:.1f}mm for {stall} ticks"
                        )
                        break
                else:
                    stall = 0
                prev_axis = cur_axis
        end = self._eef_now()
        if gentle and not outcome.note:
            outcome.note = (
                f"gentle carry ({cap * 100:.1f}cm/tick, "
                f"{align:.2f} along the opening axis)"
            )
        outcome.eef_end = list(end)
        outcome.measured_delta_m = list(np.round(end - np.asarray(outcome.eef_start), 4))
        return outcome

    # --------------------------------------------------------------- rotate
    def rotate(self, axis: str, angle_deg: float) -> ActionOutcome:
        vec = np.asarray(AXIS_VECTORS[axis], dtype=np.float64)
        # Clamp to half a full-turn budget per call; lets "turn the gripper
        # 90 deg" be a single tool call spanning several ticks.
        angle_deg = float(np.clip(angle_deg, -90.0, 90.0))
        quat_start = self._quat_now()
        outcome = ActionOutcome(
            kind="rotate",
            requested={"axis": axis, "angle_deg": angle_deg},
            eef_start=list(self._eef_now()),
        )
        with self._renderless():
            for _ in range(ROTATE_MAX_STEPS):
                done_deg = self._progress_deg(quat_start, self._quat_now(), vec)
                remaining = angle_deg - done_deg
                if abs(remaining) < ROTATE_ARRIVE_TOL_DEG:
                    break
                step_deg = math.copysign(min(ROTATE_STEP_DEG, abs(remaining)), remaining)
                action = np.zeros(C.ACTION_DIM)
                action[3:6] = vec * (math.radians(step_deg) / C.MAX_ROT_STEP_RAD)
                action[6] = C.GRIPPER_HOLD
                self.sim.step(action)
                outcome.steps += 1

        quat_end = self._quat_now()
        measured_deg = self._progress_deg(quat_start, quat_end, vec)
        outcome.eef_end = list(self._eef_now())
        outcome.measured_delta_m = list(
            np.round(np.asarray(outcome.eef_end) - np.asarray(outcome.eef_start), 4)
        )
        outcome.note = f"measured rotation about {axis}: {measured_deg:+.1f} deg"
        return outcome

    # -------------------------------------------------------------- gripper
    def set_gripper(self, action_name: str) -> ActionOutcome:
        sign = C.GRIPPER_OPEN if action_name == "open" else C.GRIPPER_CLOSE
        outcome = ActionOutcome(
            kind="set_gripper",
            requested={"action": action_name},
            eef_start=list(self._eef_now()),
        )
        prev = self._gripper_now()
        stable = 0
        with self._renderless():
            for _ in range(GRIPPER_MAX_STEPS):
                action = np.zeros(C.ACTION_DIM)
                action[6] = sign
                self.sim.step(action)
                outcome.steps += 1
                cur = self._gripper_now()
                if action_name == "open":
                    # Opening must reach a genuinely OPEN pad state. The
                    # "did it stop moving" test is WRONG here: while the pads
                    # still press on an object, the commanded action has to ramp
                    # from +1 through 0, so the width barely changes for the
                    # first few ticks (measured: 0.0234 -> 0.0242 over 5 ticks,
                    # both below the 0.002/tick convergence epsilon). The old
                    # check declared success at tick 5 with width 0.0248, so the
                    # can was NEVER RELEASED -- that is why declare_done failed
                    # twice in the v3 episode while the video looked correct
                    # (the hand carried the can down to the rim and back up).
                    if cur >= GRIPPER_OPEN_WIDTH:
                        break
                else:
                    # Closing legitimately stops when the pads meet the object
                    # (or each other), so "stopped moving" is the right signal.
                    if abs(cur - prev) < GRIPPER_CONVERGE_EPS:
                        stable += 1
                        if stable >= GRIPPER_SETTLE_AVG:
                            break
                    else:
                        stable = 0
                prev = cur

            # Clamp-force ramp (measured the hard way): the gripper command is
            # RATE-LIMITED (current_action moves 0.01/tick). Fingers stop moving
            # as soon as the pads touch the object, which trips the convergence
            # check with current_action still near 0.05 — the can was then held
            # so weakly that it SLIPPED OUT MID-CARRY and fell on the table
            # halfway to the basket. Keep commanding `close` after convergence
            # so the clamp reaches full force before we move.
            if action_name == "close":
                for _ in range(GRIPPER_RAMP_EXTRA_STEPS):
                    action = np.zeros(C.ACTION_DIM)
                    action[6] = C.GRIPPER_CLOSE
                    self.sim.step(action)
                    outcome.steps += 1

        outcome.eef_end = list(self._eef_now())
        outcome.measured_delta_m = list(
            np.round(np.asarray(outcome.eef_end) - np.asarray(outcome.eef_start), 4)
        )
        width = self._gripper_now()
        if width <= GRIPPER_EMPTY_WIDTH:
            grip_state = "empty"
        elif width >= GRIPPER_OPEN_WIDTH:
            grip_state = "open"
        else:
            grip_state = "holding"
        outcome.note = f"gripper finger width {width:.4f} ({grip_state})"
        return outcome

    # -------------------------------------------------------------- align_xy
    def align_xy(self, target_xy: tuple[float, float], z_safe: float | None = None) -> ActionOutcome:
        """Servo the EEF's x/y onto a target, holding (or raising to) a safe z.

        Added after the first real LLM episodes: the model cannot do the
        sub-centimetre arithmetic needed to line up over a can — it closed its
        fingers with 4.6 cm of x error and grasped air three times in a row
        (measured: close width 0.0013 vs 0.0197 when a can is between the
        fingers). Precise alignment is exactly the kind of atomic operation
        that belongs in the adapter, with the LLM deciding *when* and *what*.

        Alignment tolerance is the same 2 mm the validated dry-run used;
        DESCEND order is x then y (never dragging sideways into the object).
        """
        outcome = ActionOutcome(
            kind="align_xy",
            requested={"target_xy": [round(target_xy[0], 4), round(target_xy[1], 4)]},
            eef_start=list(self._eef_now()),
        )
        with self._renderless():
            if z_safe is not None and self._eef_now()[2] < z_safe:
                self._move_axis_z(z_safe, outcome)
            for _ in range(2):  # two passes: x/y are coupled by a few mm
                moved = False
                for axis_index in (0, 1):
                    for _ in range(MOVE_MAX_STEPS):
                        err = target_xy[axis_index] - self._eef_now()[axis_index]
                        if abs(err) < MOVE_ARRIVE_TOL_M:
                            break
                        step = max(-C.MAX_POS_STEP_M, min(C.MAX_POS_STEP_M, err))
                        action = np.zeros(C.ACTION_DIM)
                        action[axis_index] = step / C.MAX_POS_STEP_M
                        action[6] = C.GRIPPER_HOLD
                        self.sim.step(action)
                        outcome.steps += 1
                        moved = True
                if not moved:
                    break

        end = self._eef_now()
        outcome.eef_end = list(end)
        outcome.measured_delta_m = list(np.round(end - np.asarray(outcome.eef_start), 4))
        xy_err = float(np.linalg.norm(end[:2] - np.asarray(target_xy)))
        outcome.note = f"xy error after align: {xy_err * 1000:.1f} mm"
        return outcome

    def _move_axis_z(self, z_target: float, outcome: ActionOutcome) -> None:
        for _ in range(MOVE_MAX_STEPS):
            err = z_target - self._eef_now()[2]
            if abs(err) < MOVE_ARRIVE_TOL_M:
                break
            step = max(-C.MAX_POS_STEP_M, min(C.MAX_POS_STEP_M, err))
            action = np.zeros(C.ACTION_DIM)
            action[2] = step / C.MAX_POS_STEP_M
            action[6] = C.GRIPPER_HOLD
            self.sim.step(action)
            outcome.steps += 1

    # -------------------------------------------------------------- descend
    # Z arrival tolerance is 5 mm, NOT the 2 mm used for open-space moves:
    # against resistance (can top, table) OSC settles with a 1-3 mm standing
    # error, so a 2 mm window is never satisfied and the loop keeps commanding
    # -z — measured: 1046 ticks of pressing, which shoved the can 4 cm away and
    # then closed on air. probe_grasp_height.py (the validated sweep) used 5 mm.
    DESCEND_Z_TOL_M = 0.005

    def descend_to(
        self, xy_target: tuple[float, float], z_target: float, tol_xy: float = MOVE_ARRIVE_TOL_M
    ) -> ActionOutcome:
        """Servo to (x, y, z) with xy corrected before every descent step.

        Why a dedicated primitive (measured twice):
          * `align_xy` + a separate `move -z` leaves the descent uncorrected;
            OSC couples a few mm of lateral drift into every descent, and at the
            can's 3.3 cm radius that is enough for a finger pad to land on the
            rim — the descent then blocks at can-top height (eef ~0.087) and a
            close reads 0.0012 (air) instead of 0.0197 (can between fingers).
          * the validated sweep (probe_grasp_height.py) reached the can exactly
            because its go_to re-corrected XY after every descent step.

        This reuses the same structure: xy checked FIRST each iteration, each
        correction delegated to `move()` (its own closed loop), with a stuck
        detector so an unreachable target bails instead of burning ticks.
        """
        start = self._eef_now()
        outcome = ActionOutcome(
            kind="descend_to",
            requested={
                "target_xy": [round(xy_target[0], 4), round(xy_target[1], 4)],
                "z_target": round(z_target, 4),
            },
            eef_start=list(start),
        )
        last: list[float] | None = None
        still = 0
        for _ in range(80):
            x, y, z = self._eef_now()
            if last is not None:
                # stuck detector (probe_grasp_height.py): no progress for six
                # iterations means the target is unreachable (usually blocked).
                if abs(x - last[0]) + abs(y - last[1]) + abs(z - last[2]) < 0.001:
                    still += 1
                    if still >= 6:
                        outcome.note = f"stuck at ({x:.4f}, {y:.4f}, {z:.4f})"
                        break
                else:
                    still = 0
            last = [x, y, z]

            dx, dy, dz = xy_target[0] - x, xy_target[1] - y, z_target - z
            if abs(dx) > tol_xy:
                outcome.steps += self.move("+x" if dx > 0 else "-x", min(5.0, abs(dx) * 100)).steps
            elif abs(dy) > tol_xy:
                outcome.steps += self.move("+y" if dy > 0 else "-y", min(5.0, abs(dy) * 100)).steps
            elif abs(dz) > self.DESCEND_Z_TOL_M:
                outcome.steps += self.move("+z" if dz > 0 else "-z", min(5.0, abs(dz) * 100)).steps
            else:
                break

        end = self._eef_now()
        outcome.eef_end = list(end)
        outcome.measured_delta_m = list(np.round(end - np.asarray(outcome.eef_start), 4))
        xy_err = float(np.linalg.norm(end[:2] - np.asarray(xy_target)))
        z_err = abs(end[2] - z_target)
        if not outcome.note:
            outcome.note = f"xy err {xy_err * 1000:.1f} mm, z err {z_err * 1000:.1f} mm"
        return outcome

    # ------------------------------------------------------------- accessors
    def _eef_now(self) -> np.ndarray:
        assert self.sim._obs is not None
        return np.asarray(self.sim._obs[C.OBS_EEF_POS])

    def _quat_now(self) -> np.ndarray:
        assert self.sim._obs is not None
        return np.asarray(self.sim._obs[C.OBS_EEF_QUAT])

    def _gripper_now(self) -> float:
        """Mean absolute finger position; 0.0388 = open, ~0 = closed."""
        assert self.sim._obs is not None
        return float(np.abs(np.asarray(self.sim._obs[C.OBS_GRIPPER_QPOS])).mean())

    def _carrying(self) -> bool:
        """True when the pads hold something (not open, not shut on air)."""
        width = self._gripper_now()
        return GRIPPER_EMPTY_WIDTH < width < GRIPPER_OPEN_WIDTH

    def _opening_axis(self) -> np.ndarray | None:
        """Gripper opening axis in WORLD coordinates (EEF body y, from the quat)."""
        x, y, z, w = self._quat_now()
        col = np.array(
            [
                2.0 * (x * y - z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + x * w),
            ]
        )
        norm = float(np.linalg.norm(col))
        return col / norm if norm > 1e-9 else None

    @staticmethod
    def _progress_deg(quat_start: np.ndarray, quat_now: np.ndarray, axis: np.ndarray) -> float:
        """Signed rotation (deg) accumulated around `axis` between two quats.

        The relative rotation is q_rel = q_now * conj(q_start); decompose it to
        axis-angle and project the axis onto the commanded one. Quats are
        (x, y, z, w) as reported by robosuite.
        """
        x0, y0, z0, w0 = quat_start
        x1, y1, z1, w1 = quat_now
        # q_rel = q_now * conj(q_start)
        w = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
        x = -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0
        y = -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0
        z = -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < 1e-9:
            return 0.0
        angle = 2.0 * math.atan2(norm, w)
        if angle > math.pi:
            angle -= 2 * math.pi
        unit = np.array([x, y, z]) / norm
        signed = angle * float(np.sign(np.dot(unit, axis))) if abs(angle) > 1e-6 else 0.0
        return math.degrees(signed)
