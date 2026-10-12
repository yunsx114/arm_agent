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
import re
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
# Short action block (档 B v2). The full servo is allowed DESCEND_MAX_ITERS
# corrections; a `target` call from the model is capped at SHORT_BLOCK_ITERS so the
# arm can never run away from a plan it made without looking again (Jev bounds the
# same thing with `max_motor_steps`, 1..8, and re-plans after every block).
#
# Default = the max, so `max_steps` is purely "be more cautious than usual": the
# model asks for 1..3 when it is close and delicate, and otherwise gets one bounded
# block. Measured motivation for ANY bound: the two real pixel runs spent 36 and 19
# turns in blind `move`s after their one pixel call, and the old cap here was 80
# corrections (~40 cm of travel) in a single call.
DESCEND_MAX_ITERS = 80
SHORT_BLOCK_ITERS = 8
SHORT_BLOCK_MAX_ITERS = 8
GRIPPER_SETTLE_AVG = 5  # finger must move < eps per step for this many steps
GRIPPER_MAX_STEPS = 130  # measured: full sweep needs ~100 steps at 0.01/step
# Extra full-power ticks after the pads stop moving on an object. The clamp
# command is rate-limited (0.01/tick) so convergence fires with low force, but
# a long ramp (measured: 100 ticks) SQUEEZES THE CAN OUT of the grip
# (close width 0.0005 afterwards) and 30 ticks let it slip mid-carry.
# 10 ticks firms the pinch without ejecting.
GRIPPER_RAMP_EXTRA_STEPS = 10
# ---- grasp / release concepts (档 A, 2026-10-11) --------------------------
# Contact is decided from the raw MuJoCo contact table, never from the finger
# width. MEASURED with scripts/probe_contact_table.py: at reset the pads sit
# HALF CLOSED (width 0.0207) inside the old "something is between the pads"
# band while the contact table holds ZERO finger contacts -- so a width test
# calls the boot state a grasp. Closing on air walks through the same band.
CONTACT_DIST_TOL_M = 0.0005  # MuJoCo reports ~-0.1..-0.35 mm for real pad contact
# Extra closed-loop ticks after bilateral contact, so the clamp reaches full
# force before anything moves (the ramp above is the same idea at close time).
GRASP_SQUEEZE_TICKS = 10
# Bilateral contact must persist this long before the grasp counts as secured;
# losing a side inside the window is reported as `slipped`.
GRASP_HOLD_TICKS = 10
# After opening, wait up to this many ticks for the pads to clear the payload.
# MEASURED (8/8 episodes, 2026-10-11): `verify` immediately after an `open`
# fails EVERY time, because the pads are still opening while the payload has not
# started falling -- a systematic false negative.
RELEASE_SETTLE_TICKS = 15

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
    # --- concept layer (档 A) -------------------------------------------
    # `status` is the closed-vocabulary verdict the MODEL is shown
    # ("grasped"/"empty"/"slipped"/"released"/"still_carrying"/"").
    # `measurement` is the same event in numbers and stays in the transcript:
    # the model gets the concept, the instruments keep the resolution.
    status: str = ""
    measurement: dict[str, Any] = field(default_factory=dict)
    # None = this action says nothing about what is between the pads.
    held: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "requested": self.requested,
            "steps": self.steps,
            "eef_start": self.eef_start,
            "eef_end": self.eef_end,
            "measured_delta_m": self.measured_delta_m,
            "note": self.note,
            "status": self.status,
            "measurement": self.measurement,
            "held": self.held,
        }


def classify_descent(outcome: ActionOutcome) -> str:
    """One vocabulary for "did the descent get there": arrived / short / blocked.

    `descend_to` writes the note (`xy err .. mm, z err .. mm`, or `stuck at (..)`)
    and this is the ONLY place that interprets it. Two consumers read it -- the
    legacy `approach` text path and the 档 B `goto_pixel` path -- and the P0f bug
    (a key from one vocabulary compared against a value from another, with the
    warning firing 0 times across two runs) is exactly what happens when two
    places classify the same note independently.
    """
    note = str(outcome.note or "")
    if "stuck" in note:
        return "blocked"
    match = re.search(r"z err ([0-9.]+) mm", note)
    if match is None:
        return "arrived" if outcome.steps else "short"
    return "arrived" if float(match.group(1)) <= ActionAdapter.DESCEND_Z_TOL_M * 1000.0 else "short"


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

    # --------------------------------------------------------- grasp concept
    def _pad_sides(self) -> tuple[set[int], set[int], list[str]]:
        """Finger-pad body ids as (left, right, body_names).

        MEASURED (scripts/probe_contact_table.py, 2026-10-11, LIBERO object task
        0): the bodies that actually touch a payload are
        `gripper0_finger_joint1_tip` (left) and `gripper0_finger_joint2_tip`
        (right). The finger *base* bodies (`gripper0_leftfinger`,
        `gripper0_rightfinger`) never appear in the contact table at all.

        The names come from `sim.model.body_names`; the first version of that
        probe asked for `model.body_id2name`, which does not exist on the
        robosuite wrapper, and silently printed `<body N>` for every body --
        a "value from another vocabulary" bug of exactly the kind
        scripts/probe_tool_wiring.py exists to prevent.
        """
        table = list(getattr(self.sim.env.env.sim.model, "body_names", []) or [])
        left: set[int] = set()
        right: set[int] = set()
        for body_id, name in enumerate(table):
            low = str(name).lower()
            if "finger" not in low:
                continue
            if "joint1" in low or "left" in low:
                left.add(body_id)
            elif "joint2" in low or "right" in low:
                right.add(body_id)
        tips = [i for i, n in enumerate(table) if "finger" in str(n).lower() and "tip" in str(n).lower()]
        if tips:  # prefer the pads that make contact; the bases never do
            return {i for i in tips if i in left}, {i for i in tips if i in right}, table
        return left, right, table

    def pad_contacts(self) -> dict[str, set[str]]:
        """Object body name -> set of pad sides currently touching it.

        This is the scene-independent half of the grasp concept: it names no
        object, uses no coordinate and no width, and works for any scene whose
        gripper has two finger pads. `dist <= CONTACT_DIST_TOL_M` is Jev's
        filter (embodied_jev/physics.py::contacts) and matches what MuJoCo
        reports for a real pad contact here (-0.1 .. -0.35 mm).
        """
        left, right, table = self._pad_sides()
        if not left and not right:
            return {}
        sim = self.sim.env.env.sim
        model = getattr(sim.model, "_model", sim.model)
        data = sim.data
        out: dict[str, set[str]] = {}
        for contact in data.contact[: int(data.ncon)]:
            if float(contact.dist) > CONTACT_DIST_TOL_M:
                continue
            a = int(model.geom_bodyid[int(contact.geom1)])
            b = int(model.geom_bodyid[int(contact.geom2)])
            for pad, other in ((a, b), (b, a)):
                side = "left" if pad in left else "right" if pad in right else None
                if side is None:
                    continue
                name = str(table[other]) if 0 <= other < len(table) else f"<body {other}>"
                out.setdefault(name, set()).add(side)
        return out

    def held_object(self) -> tuple[str | None, set[str]]:
        """(body name between BOTH pads, the sides touching it).

        `name` is not None only for a bilateral contact -- that is the whole
        point of the concept. A single-side contact is reported as such so the
        caller can tell `slipped` from `empty`.
        """
        contacts = self.pad_contacts()
        if not contacts:
            return None, set()
        name, sides = max(contacts.items(), key=lambda kv: (len(kv[1]), kv[0]))
        return (name if len(sides) == 2 else None), sides

    def _command_gripper(self, value: float, outcome: ActionOutcome) -> None:
        """One tick of a pure gripper command (translation channels hold)."""
        action = np.zeros(C.ACTION_DIM)
        action[6] = value
        self.sim.step(action)
        outcome.steps += 1

    def _open_pads(self, outcome: ActionOutcome) -> float:
        """Command `open` until the pads are genuinely open; returns the width."""
        width = self._gripper_now()
        while width < GRIPPER_OPEN_WIDTH and outcome.steps < GRIPPER_MAX_STEPS:
            self._command_gripper(C.GRIPPER_OPEN, outcome)
            width = self._gripper_now()
        return width

    def grasp(self) -> ActionOutcome:
        """Close the pads around whatever is between them and report a CONCEPT.

        WHY this is a primitive and the width is not: the finger width is not a
        grasp. MEASURED at reset (probe_contact_table.py): the sim boots with the
        pads HALF CLOSED (width 0.0207) inside the "something is between the
        pads" band while the contact table holds ZERO finger contacts, and
        closing on air walks through the same band. So a width test calls the
        boot state -- and every empty close -- a grasp, which is exactly the
        false positive that cost an episode earlier (P0a: 7 "slip" + 5 "dropped"
        alarms, and a can shed mid-carry).

        The sequence is the one the project measured by hand, now owned by code:
          1. OPEN first. The sim starts half-closed, so a bare `close` only
             pinches (~0.0183) -- the documented cause of a "looks grasped,
             sheds on the first lateral move" failure.
          2. Close until BOTH pads contact the same body (the concept), or the
             pads shut on air.
          3. Keep commanding `close` for GRASP_SQUEEZE_TICKS so the clamp reaches
             full force; the command is rate-limited (0.01/tick), so stopping at
             first contact leaves a weak grip (measured: the can slid out
             mid-carry) while a 100-tick ramp squeezed it out the other side.
          4. Hold and re-check for GRASP_HOLD_TICKS: contact that disappears
             inside the window is a `slipped`, not a `grasped`.
        The model is told the concept; the numbers stay in `measurement`.
        """
        outcome = ActionOutcome(
            kind="grasp",
            requested={"action": "grasp"},
            eef_start=list(self._eef_now()),
        )
        with self._renderless():
            open_width = self._open_pads(outcome)
            # --- close until bilateral contact (or a shut pad pair) ----------
            secured = False
            sides: set[str] = set()
            body: str | None = None
            prev = self._gripper_now()
            stable = 0
            while outcome.steps < GRIPPER_MAX_STEPS:
                self._command_gripper(C.GRIPPER_CLOSE, outcome)
                body, sides = self.held_object()
                if len(sides) == 2:
                    secured = True
                    break
                cur = self._gripper_now()
                if cur <= GRIPPER_EMPTY_WIDTH:
                    break  # shut on air
                if abs(cur - prev) < GRIPPER_CONVERGE_EPS:
                    stable += 1
                    if stable >= GRIPPER_SETTLE_AVG:
                        break  # converged without a bilateral contact
                else:
                    stable = 0
                prev = cur

            squeeze_ticks = hold_ticks = 0
            lost_side = False
            if secured:
                for _ in range(GRASP_SQUEEZE_TICKS):
                    self._command_gripper(C.GRIPPER_CLOSE, outcome)
                    squeeze_ticks += 1
                for _ in range(GRASP_HOLD_TICKS):
                    self._command_gripper(C.GRIPPER_HOLD, outcome)
                    hold_ticks += 1
                    _, now_sides = self.held_object()
                    if len(now_sides) < 2:
                        lost_side = True
                        sides = now_sides
                        break

        width = self._gripper_now()
        if secured and not lost_side:
            status = "grasped"
        elif secured and lost_side:
            status = "slipped"
        elif sides:
            status = "slipped" if len(sides) == 1 else "empty"
        else:
            status = "empty"
        outcome.status = status
        outcome.held = status == "grasped"
        outcome.note = f"grasp {status}"
        outcome.measurement = {
            "object_body": body,
            "contact_sides": sorted(sides),
            "width_mm": round(width * 1000, 2),
            "open_width_mm": round(open_width * 1000, 2),
            "squeeze_ticks": squeeze_ticks,
            "hold_ticks": hold_ticks,
        }
        return outcome

    def release(self) -> ActionOutcome:
        """Open the pads, WAIT for the payload to clear them, report a CONCEPT.

        The settle window is not cosmetic. MEASURED (8/8 episodes, 2026-10-11):
        `verify` issued immediately after `open` fails EVERY time -- the pads are
        still opening and the payload has not begun to fall, so the
        environment's "is it placed" check is answered too early, and the model
        read that as "my release failed" and dived back into the basket (in one
        case dragging the can 29 cm back out of it). Returning only after the
        pads have actually cleared the object makes the next `verify` honest.

        No scene knowledge: the concept is "is anything between the pads", the
        same contact table `grasp` uses.
        """
        outcome = ActionOutcome(
            kind="release",
            requested={"action": "release"},
            eef_start=list(self._eef_now()),
        )
        with self._renderless():
            width = self._open_pads(outcome)
            cleared_after = None
            for tick in range(RELEASE_SETTLE_TICKS):
                self._command_gripper(C.GRIPPER_HOLD, outcome)
                body, sides = self.held_object()
                if not sides:
                    cleared_after = tick + 1
                    break
                width = self._gripper_now()

        body, sides = self.held_object()
        still = bool(sides)
        outcome.status = "still_carrying" if still else "released"
        outcome.held = still
        outcome.note = f"release {outcome.status}"
        outcome.measurement = {
            "object_body": body,
            "contact_sides": sorted(sides),
            "width_mm": round(width * 1000, 2),
            "cleared_after_ticks": cleared_after,
        }
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
        self,
        xy_target: tuple[float, float],
        z_target: float,
        tol_xy: float = MOVE_ARRIVE_TOL_M,
        max_iters: int = DESCEND_MAX_ITERS,
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

        `max_iters` is the SHORT ACTION BLOCK (档 B v2, copied from Jev's
        `max_motor_steps`): the caller can cut the servo off after N corrections and
        make the model look again. One iteration = one xy/z correction, each of which
        is its own closed loop of physics ticks, so this is a coarse-grained knob
        (Jev's 1..8 range maps onto it), not a tick budget. When the block runs out
        before convergence the outcome says so (`measurement.truncated`) instead of
        pretending the shortfall was a blocked path.
        """
        limit = max(1, int(max_iters))
        start = self._eef_now()
        outcome = ActionOutcome(
            kind="descend_to",
            requested={
                "target_xy": [round(xy_target[0], 4), round(xy_target[1], 4)],
                "z_target": round(z_target, 4),
                "max_iters": limit,
            },
            eef_start=list(start),
        )
        last: list[float] | None = None
        still = 0
        converged = False
        iters = 0
        for _ in range(limit):
            iters += 1
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
                converged = True
                break

        end = self._eef_now()
        outcome.eef_end = list(end)
        outcome.measured_delta_m = list(np.round(end - np.asarray(outcome.eef_start), 4))
        xy_err = float(np.linalg.norm(end[:2] - np.asarray(xy_target)))
        z_err = abs(end[2] - z_target)
        if not outcome.note:
            outcome.note = f"xy err {xy_err * 1000:.1f} mm, z err {z_err * 1000:.1f} mm"
        # Two different shortfalls, two different sentences. "stuck" (unreachable)
        # and "out of block" (the caller asked for a short move) both leave the arm
        # short of the target, and the model is the one that has to tell them apart.
        # Append AFTER the note, never instead of it: `classify_descent` parses
        # "z err N mm" and "stuck" out of this string.
        truncated = not converged and iters >= limit and "stuck" not in outcome.note
        if truncated:
            outcome.note += f" | 短块 {iters} 步用尽，未收敛"
        outcome.measurement.update({"iterations": iters, "truncated": bool(truncated)})
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
