"""Canonical tool protocol and structured result helpers.

The simulator still accepts the legacy action vocabulary. This module defines
the vocabulary exposed to the model and keeps protocol validation independent
from the simulator implementation.
"""

from __future__ import annotations

from typing import Any

CANONICAL_TOOLS = (
    "observe", "approach", "goto_pixel", "move", "rotate", "gripper", "grasp", "release", "verify",
)
# Tool surfaces that are legal in ONE observation mode only (档 B). The mode is
# owned by the simulator, so the harness mirrors it here purely to keep the
# advertised schema honest: in `pixels` mode `approach` would have to read the
# simulator's own object positions, which is exactly what that mode removes.
MODE_ONLY_TOOLS: dict[str, frozenset[str]] = {
    "pixels": frozenset({"approach"}),
}
# Registered here but blocked under v2 by UNREGISTERED_TOOLS below.
# Temporarily UNREGISTERED surfaces: the code path stays (dispatch, aliases,
# validation) but the model is neither shown the tool nor allowed to call it.
#
# `observe` is off since 2026-10-10 on purpose. Every turn now carries BOTH
# camera frames, so the question this ablation answers is "can the model work
# from pixels alone?" -- previously `observe <target>` was the ONLY source of
# object coordinates, and it was also the only way to refresh a frame.
# Re-enabling is deleting the entry below.
#
# `gripper` (open/close + a finger-width readout) is off since 2026-10-11: it is
# replaced by `grasp` / `release`, which decide from the CONTACT table and answer
# in concepts instead of millimetres. Blocking it under v2 means the ablation
# cannot silently fall back to the old surface -- the same reason `observe` is
# blocked, and the validator checks this (see UNREGISTERED_TOOLS below).
#
# The validator check matters for experimental validity: a model that calls a
# tool from memory would otherwise still succeed, silently leaking the numbers
# this ablation is trying to remove.
UNREGISTERED_TOOLS: frozenset[str] = frozenset({"observe", "gripper"})
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
}

REQUIRED = {
    "observe": (),
    "approach": ("target",),
    # `goto_pixel` requires the pixel and nothing else: the clearance offset is
    # optional because 0 is a legitimate request ("ground this pixel and go")
    # and the camera defaults to the third-person view.
    "goto_pixel": ("u", "v"),
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
    "goto_pixel": {"camera": {"agent", "wrist"}},
    "move": {"direction": {"+x", "-x", "+y", "-y", "+z", "-z"}},
    "rotate": {"axis": {"yaw", "pitch", "roll"}},
    "gripper": {"action": {"open", "close"}},
    "verify": {"claim": {"placed", "grasped", "task_complete"}},
}


def canonical_name(name: str) -> str:
    return ALIASES.get(name, name)


def validate_tool_call(
    name: str, args: dict[str, str], *, legacy: bool = False, observation_mode: str = "truth"
) -> list[str]:
    """Return explicit protocol errors; do not execute invalid calls."""
    allowed = LEGACY_TOOLS if legacy else CANONICAL_TOOLS
    if name not in allowed:
        return [f"unknown_tool:{name}"]
    # Mode-excluded tools are rejected HERE rather than silently dispatched, so a
    # `pixels` run cannot answer a question it claims not to have the data for.
    blocked_in_mode = MODE_ONLY_TOOLS.get(str(observation_mode).strip().lower(), frozenset())
    if not legacy and (canonical_name(name) in blocked_in_mode or name in blocked_in_mode):
        return [f"mode_blocked_tool:{name}"]
    # Scoped to the v2 surface on purpose: the legacy surface is the documented
    # A/B path (and the no-GPU `probe_harness_dryrun.py` regression), which must
    # keep working while this ablation is on. The ablation itself cannot leak
    # through legacy names -- `look`/`locate`/`get_state` are not advertised
    # anywhere in a v2 conversation, so the model has no reason to emit them.
    if not legacy and (name in UNREGISTERED_TOOLS or canonical_name(name) in UNREGISTERED_TOOLS):
        return [f"unregistered_tool:{name}"]
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
