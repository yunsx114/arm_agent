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
