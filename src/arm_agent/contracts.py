"""Shared observation/action contracts for the simulation side.

Mirrors the role of a3_dual_arm_sim's contracts.py: one place that names the
strings and numbers the whole sim+agent stack agrees on, so neither side
scatters magic values.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Action space (robosuite OSC_POSE, 7 dims)
# ---------------------------------------------------------------------------
ACTION_DIM = 7

# Measured from `load_controller_config(default_controller="OSC_POSE")` in the
# libero env (2026-10-07): ./probe/controller_config. input range [-1, 1].
OSC_OUTPUT_MAX = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)  # meters, then radians
MAX_POS_STEP_M = OSC_OUTPUT_MAX[0]  # 5 cm per env.step()
MAX_ROT_STEP_RAD = OSC_OUTPUT_MAX[3]  # 0.5 rad = 28.6 deg per env.step()

# Axis-angle rotation deltas are applied as goal = R_err @ R_current (left
# multiply, verified in robosuite/utils/control_utils.py::set_goal_orientation),
# i.e. they are expressed in the BASE (world) frame. Same for position deltas
# (goal_position = current_position + delta).

# Gripper: robosuite/models/grippers/panda_gripper.py::format_action docstring:
#   "-1 => open, 1 => closed"
# and it is rate-limited: current_action moves by +/-0.01 per sim step, so a
# full open->close sweep takes ~100 steps (5 s at control_freq=20). The adapter
# therefore holds the command until the fingers actually converge.
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0
GRIPPER_HOLD = 0.0  # sign(0)=0 -> current_action unchanged (keeps last state)
GRIPPER_OPEN_QPOS = 0.0208  # measured initial finger position (rad)
GRIPPER_CLOSED_QPOS = 0.0

# ---------------------------------------------------------------------------
# Observation keys (LIBERO OffScreenRenderEnv, 40 keys on libero_object task 0)
# ---------------------------------------------------------------------------
OBS_AGENTVIEW = "agentview_image"  # 256x256x3 uint8, display-oriented (flip v)
OBS_WRIST = "robot0_eye_in_hand_image"
# MuJoCo camera names (for calibration / depth rendering), keyed by the name the
# MODEL uses in a tool call. 档 B needs the mapping in one place: the tool speaks
# "agent"/"wrist" while the simulator speaks "agentview"/"robot0_eye_in_hand".
CAMERA_NAMES = {"agent": "agentview", "wrist": "robot0_eye_in_hand"}

# 档 E (planner/executor split): the one cross-model argument the protocol needs.
# The sub-model's TURN budget -- after it, the sub-episode returns control no matter
# what (the user's spec: "预算结束后自动返回"). The default lives with the schema
# description; the range is checked by `validate_tool_call`.
# MEASURED (planner run 2026-10-12, 8 episodes / 162 dispatches): the executor used
# 100% of its budget in 139/162 dispatches and NEVER returned early -- it has the
# `verify` exit available and used it zero times -- so a phase is not "cut short by a
# lazy sub", it is cut short by US. Every phase therefore has to be sized for the whole
# phase: the planner's 8- and 10-turn carry phases dropped the can mid-air, while the
# 45-turn grasp phase in episode 3 succeeded. Floor raised from 5 to 20 so an
# unfinishable phase is rejected at validation time instead of at the arm.
DISPATCH_BUDGET_RANGE = (20, 80)
DISPATCH_BUDGET_DEFAULT = 45
# 档 F: cap on a SINGLE downward move when the model (not the ladder) names the
# centimetres. MEASURED (planner+high, 2026-10-15): the one surviving failure mode of
# all three lost episodes was a -z request of 6~15 cm in one shot -- the arm sailed
# past the "stop at z~0.07" instruction to z=0.0367 (table), closed on air and knocked
# the can over, and every episode then spent itself chasing the lying-down can. The
# winning episode descended in ≤3 cm steps. Upward moves stay uncapped (17 cm lifts
# were used all run without incident); horizontal is untouched (visual closed loop).

# The direction is the MODEL's decision and is preserved (same philosophy as the
# ladder: scale the magnitude, never replace the direction); the clamp is announced
# in the tool text so the executor knows the step was shortened and can continue.
DESCENT_CAP_CM = 3.0
# Key of the magnified crop in an image payload (档 B v2 `aim`/`target`). It lives
# here because BOTH sides need it and neither may import the other: the agent process
# has no robosuite, and the sim process has no model client. It is not a camera, so it
# must never collide with a key of CAMERA_NAMES.
ZOOM_KEY = "zoom"

# MOVE MAGNITUDE LADDER (档 D). When it is on, `move` takes one of three NAMED rungs
# instead of a distance in centimetres, so the model decides a DIRECTION and a
# GRANULARITY while the magnitude stays code-owned -- Jev's "scale magnitudes, never
# replace a model-selected direction" (embodied_jev/hierarchical.py::assemble_motion,
# whose own menu is 2/10/40 mm).
#
# WHY: measured, the model cannot do the arithmetic this replaces. In `pixels` mode its
# absolute pixel reading carries a 30-50 px systematic bias (10-13 cm); in `raw` mode
# it estimated the can at y=-0.10 when the truth is y=-0.24, i.e. a 14 cm error. Both
# are the same missing ability -- turning what it sees into a world distance. Three
# rungs also give it a natural closed loop: overshoot -> downshift, too little ->
# upshift, with no numbers to get wrong.
#
# The intended use is `far` to cross the workspace, `near` to line up, `fine` to trim.
# 1 mm is a REAL rung: `action_adapter.move` scales its arrival tolerance with the
# request (`arrive_tol = min(MOVE_ARRIVE_TOL_M, request/2)`), which was added after a
# 1 mm request was swallowed by the old fixed 2 mm window -- it ran 0 ticks and
# reported a (0,0,0) displacement that looked exactly like "blocked".
MOVE_LADDER_CM: dict[str, float] = {"far": 10.0, "near": 1.0, "fine": 0.1}
# Literal spellings accepted as the same rung, because a schema enum is easier to
# follow than a vocabulary to memorise -- and getting the word wrong costs a turn.
MOVE_LADDER_ALIASES: dict[str, str] = {
    "10cm": "far", "1cm": "near", "1mm": "fine",
    "coarse": "far", "medium": "near", "mid": "near", "small": "fine", "tiny": "fine",
}


def move_rung_cm(size: str) -> float | None:
    """Centimetres for a ladder rung (or None if the token is not a rung)."""
    token = str(size).strip().lower()
    rung = MOVE_LADDER_ALIASES.get(token, token)
    return MOVE_LADDER_CM.get(rung)


# PIXEL-GAP MODE (档 D v2). MEASURED (three arms, 24 episodes, transcripts): the model
# READS pixels acceptably (its quoted gap runs 1.2-3.2x the truth) but its px->cm
# MULTIPLICATION is off by ~11x -- it writes "75px x 3.8mm = 28.5mm = 2.9cm", i.e. it
# knows the scale and then drops a factor of ten (mm vs cm). That single slip is ~4/5 of
# its total distance error, and it makes every arm pick a step that is too small: a 10-19cm
# gap is believed to be 2-3cm, so it creeps at 1cm and never arrives in 80 turns.
#
# So the model reports the thing it can actually see -- a pixel gap along one axis -- and
# the CODE, which knows the camera exactly, does the arithmetic and snaps the result to a
# rung. Magnitudes stay code-owned (the ladder), directions and pixel readings stay the
# model's (the two things it does well). Default scale: the MEASURED 2.96-4.04 mm/px of the
# agent view; it self-calibrates from every `mm_per_px` the simulator reports back.
GLOBAL_MM_PER_PX_DEFAULT = 3.7
# Which rung a pixel gap maps to. Thresholds are the geometric means of the rungs, so the
# snap is scale-free: 0.32cm..3.2cm -> near, >=3.2cm -> far, <0.32cm -> fine.
MOVE_LADDER_SNAP_CM = (0.32, 3.2)


def rung_for_gap_cm(gap_cm: float, *, mm_per_px: float | None = None) -> tuple[str, float]:
    """(rung, centimetres) for a measured gap: the CLOSEST rung on a log scale.

    Thresholds are the geometric means of the rungs (sqrt(0.1*1)=0.32, sqrt(1*10)=3.16),
    which is what makes this converge fastest: the error shrinks by at most 10^0.5 = 3.16x
    per step, so ~30cm reaches 1cm in three steps and 1mm in six, whatever the starting
    point. "Largest rung that does not overshoot" was the first instinct and it is wrong
    here: it turns a 9cm gap into nine 1cm steps, and a turn budget is the one resource
    the measured runs run out of first (119 turns for a scripted full-knowledge controller).
    Overshoot is handled the way the prompt already says -- downshift after a reversal --
    and every step now reports the pixels it actually moved, so an overshoot is visible.
    """
    del mm_per_px  # kept in the signature for callers that want to log the scale used
    fine_hi, near_hi = MOVE_LADDER_SNAP_CM
    if gap_cm >= near_hi:
        return "far", MOVE_LADDER_CM["far"]
    if gap_cm >= fine_hi:
        return "near", MOVE_LADDER_CM["near"]
    return "fine", MOVE_LADDER_CM["fine"]
OBS_EEF_POS = "robot0_eef_pos"
OBS_EEF_QUAT = "robot0_eef_quat"
OBS_GRIPPER_QPOS = "robot0_gripper_qpos"
OBS_OBJECT_STATE = "object-state"
OBS_SETTLE_STEPS = 5  # official demo convention: 5 zero-action steps at reset

# The robot is the only key prefix excluded from GT object extraction.
ROBOT_PREFIX = "robot0_"

# ---------------------------------------------------------------------------
# IPC (NFS file-drop protocol, see sim/server.py for the measured rationale)
# ---------------------------------------------------------------------------
IPC_CMD_GLOB = "cmd_*.json"
IPC_POLL_INTERVAL_S = 0.01  # 10 ms; NFS file ops measured at ~1 ms round trip
IPC_DEFAULT_TIMEOUT_S = 600.0  # an episode's worth of slow tool execution
"""arm_agent: VLM tool-calling agent driving a LIBERO/robosuite arm."""
