"""Canonical tool protocol and structured result helpers.

The simulator still accepts the legacy action vocabulary. This module defines
the vocabulary exposed to the model and keeps protocol validation independent
from the simulator implementation.
"""

from __future__ import annotations

from typing import Any

from arm_agent import contracts as C

CANONICAL_TOOLS = (
    "observe", "aim", "target", "approach", "move", "rotate", "gripper", "grasp", "release", "verify",
)
# WHICH TOOLS EACH OBSERVATION MODE ADVERTISES AND ACCEPTS (v2).
#
# A whitelist, not a blocklist. With a blocklist every new mode silently inherits
# every tool it forgot to exclude -- which is exactly how `gripper` would have leaked
# back into `truth` after 档 A replaced it, and how a "no pixel tools" mode would have
# kept the pixel tools.
#
#   truth  -- the simulator owns object positions, so the model aligns by NAME.
#             Left byte-identical to the surface the 16-episode A/B baseline ran on.
#   pixels -- no object coordinates anywhere; the model names pixels (档 B).
#   raw    -- the minimum surface that can still do a pick-and-place: move + open/
#             close + the env check. No localisation aid of ANY kind, neither by name
#             nor by pixel -- the model has to convert what it sees into world-axis
#             motion itself. This is the last rung of the ladder: every tool removed
#             is a crutch removed.
MODE_TOOLS: dict[str, frozenset[str]] = {
    "truth": frozenset({"approach", "move", "rotate", "grasp", "release", "verify"}),
    "pixels": frozenset({"aim", "target", "move", "rotate", "grasp", "release", "verify"}),
    "raw": frozenset({"move", "gripper", "verify"}),
}
DEFAULT_MODE = "truth"
# Registered in code, advertised in NO mode: the path stays (dispatch, aliases,
# validation) but the model is neither shown the tool nor allowed to call it.
#
# `observe` has been off since 2026-10-10: every turn now carries BOTH camera frames,
# so `observe <target>` would only re-add object coordinates, which is the thing the
# observations modes exist to remove.
UNREGISTERED_TOOLS: frozenset[str] = frozenset({"observe"})


def tools_for_mode(observation_mode: str) -> frozenset[str]:
    """The v2 surface of one observation mode (unknown mode -> the truth surface)."""
    return MODE_TOOLS.get(str(observation_mode).strip().lower(), MODE_TOOLS[DEFAULT_MODE])

LEGACY_TOOLS = (
    "look", "locate", "descend_to", "align_xy", "get_state", "move",
    "rotate", "set_gripper", "declare_done", "write_skill", "read_skill",
)

ALIASES = {
    "look": "observe",
    "locate": "observe",
    "get_state": "observe",
    "descend_to": "approach",
    "align_xy": "approach",
    "set_gripper": "gripper",
    "declare_done": "verify",
    # 档 B v2 renamed `goto_pixel` to `target` (it now moves as a short block). The
    # old name stays valid so a model with the old name in its history -- and every
    # transcript, probe and skill file written before today -- still routes here
    # instead of failing as an unknown tool.
    "goto_pixel": "target",
}

REQUIRED = {
    "observe": (),
    "approach": ("target",),
    # `aim` and `target` require the pixel and nothing else: the clearance offset is
    # optional because 0 is a legitimate request ("ground this pixel and go"), the
    # camera defaults to the third-person view, and the block length defaults to the
    # short-block constant (see action_adapter.SHORT_BLOCK_ITERS).
    "aim": ("u", "v"),
    "target": ("u", "v"),
    "move": ("direction", "distance_cm"),
    "rotate": ("axis", "angle_deg"),
    "gripper": ("action",),
    # `grasp` / `release` take no parameters on purpose: how far to close, when
    # to squeeze and when the payload has cleared the pads are all decided from
    # the contact table (see action_adapter.grasp/release). A parameter here
    # would be the harness asking the model to guess millimetres again.
    "grasp": (),
    "release": (),
    # `claim` is deliberately NOT required: `declare_done` ignores it and asks
    # the environment for the ground truth, so requiring it only gave the model
    # a way to fail validation on a parameter that changes nothing.
    "verify": (),
}

ENUMS = {
    "observe": {"camera": {"agent", "wrist"}},
    # `aim` may look through EITHER camera: looking does not move the arm, and the
    # wrist view is the fine-grained one (measured 1.10 mm/px at 12 cm vs 3.0-4.0
    # from the third-person view). `target` is agent-only because an eye-in-hand
    # pixel is a DEGENERATE servo target (measured: a fixed wrist pixel is a fixed
    # point in the hand frame, so it travels with the gripper and can never be
    # reached).
    "aim": {"camera": {"agent", "wrist"}},
    "target": {"camera": {"agent"}},
    "move": {"direction": {"+x", "-x", "+y", "-y", "+z", "-z"}},
    "rotate": {"axis": {"yaw", "pitch", "roll"}},
    "gripper": {"action": {"open", "close"}},
    "verify": {"claim": {"placed", "grasped", "task_complete"}},
}


def canonical_name(name: str) -> str:
    return ALIASES.get(name, name)


def validate_tool_call(
    name: str,
    args: dict[str, str],
    *,
    legacy: bool = False,
    observation_mode: str = "truth",
    move_ladder: bool = False,
) -> list[str]:
    """Return explicit protocol errors; do not execute invalid calls."""
    allowed = LEGACY_TOOLS if legacy else CANONICAL_TOOLS
    if name not in allowed:
        return [f"unknown_tool:{name}"]
    # The MODE's surface is enforced HERE rather than silently dispatched, so a run
    # cannot answer a question its mode claims not to have the data for (e.g. a
    # `pixels` run calling `approach`, or a `raw` run calling `aim`).
    if not legacy:
        canonical = canonical_name(name)
        surface = tools_for_mode(observation_mode)
        if canonical not in surface:
            return [f"mode_blocked_tool:{name}"]
    canonical = canonical_name(name)
    normalized_args = dict(args)
    if name in ("locate", "descend_to", "align_xy"):
        normalized_args["target"] = args.get("name", "")
    if name == "declare_done":
        normalized_args["claim"] = args.get("answer_note", "task_complete")
    errors = [
        f"missing:{key}"
        for key in REQUIRED.get(canonical, ())
        if not str(normalized_args.get(key, "")).strip()
    ]
    # MOVE LADDER (档 D): with the ladder on, `move` accepts a NAMED rung and nothing
    # else. A distance in centimetres is rejected with a reason instead of being
    # silently converted -- the whole point is that the model is not doing the
    # arithmetic any more, and a silent conversion would hide it still trying to.
    if canonical == "move" and move_ladder:
        if str(normalized_args.get("distance_cm", "")).strip():
            return ["ladder_rejects:distance_cm(use gap_px or size=far|near|fine)"]
        size = str(normalized_args.get("size", "")).strip().lower()
        gap = str(normalized_args.get("gap_px", "")).strip()
        if not size and not gap:
            return ["missing:gap_px(画面里还差多少像素)"]
        if size and gap:
            return ["ambiguous:size_and_gap_px(只给一个)"]
        if size and C.move_rung_cm(size) is None:
            return [f"invalid_enum:size={size}(far|near|fine)"]
        if gap:
            try:
                gap_px = float(gap)
            except ValueError:
                return [f"invalid_number:gap_px={gap}"]
            if not 0.5 <= gap_px <= 300:
                return [f"invalid_range:gap_px={gap_px}(0.5~300像素)"]
        # The rung replaces the distance ONLY. `direction` is still the model's call
        # and still has to be one of the six axes -- returning early here without this
        # check would have silently accepted `direction="向着罐子"`.
        direction = str(normalized_args.get("direction", "")).strip()
        if not direction:
            return ["missing:direction(+x|-x|+y|-y|+z|-z)"]
        if direction not in ENUMS["move"]["direction"]:
            return [f"invalid_enum:direction={direction}"]
        return []
    for key, values in ENUMS.get(canonical, {}).items():
        value = str(normalized_args.get(key, "")).strip()
        if value and value not in values:
            errors.append(f"invalid_enum:{key}={value}")
    if canonical == "move" and args.get("distance_cm"):
        try:
            distance = float(args["distance_cm"])
            if not 0.1 <= distance <= 15:
                errors.append("out_of_range:distance_cm[0.1,15]")
        except ValueError:
            errors.append("invalid_number:distance_cm")
    if canonical == "rotate" and args.get("angle_deg"):
        try:
            angle = float(args["angle_deg"])
            if not -90 <= angle <= 90:
                errors.append("out_of_range:angle_deg[-90,90]")
        except ValueError:
            errors.append("invalid_number:angle_deg")
    if canonical == "target" and args.get("max_steps"):
        # Bounded exactly like Jev's `max_motor_steps` (1..8): the block length is a
        # knob on how often the model must look again, not a way to buy a long blind
        # run -- 36 and 19 turns of blind motion were measured with an unbounded one.
        try:
            steps = int(float(args["max_steps"]))
            if not 1 <= steps <= 8:
                errors.append("out_of_range:max_steps[1,8]")
        except (TypeError, ValueError):
            errors.append("invalid_number:max_steps")
    return errors


def structured_result(
    *,
    tool: str,
    status: str,
    requested: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the stable result envelope persisted in transcripts."""
    result: dict[str, Any] = {
        "protocol": "arm-agent-tools-v2",
        "tool": tool,
        "status": status,
        "requested": requested or {},
        "measurement": measurement or {},
        "events": events or [],
    }
    if error is not None:
        result["error"] = error
    return result
