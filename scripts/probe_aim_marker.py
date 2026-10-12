"""Dump one `aim` frame so the ring can be checked by EYE, not only by a counter.

The dry-run leg counts pixels of AIM_MARK_COLOUR, which proves something red is on the
frame but not that it is where the model asked or that the target is readable next to
it. This writes the two frames (agent + wrist) plus the ground-truth pixel of the can,
aimed deliberately 30 px off so the comparison the tool exists for is visible.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.pixel_grounding import Camera  # noqa: E402
from arm_agent.sim.server import SimServer  # noqa: E402

OFFSET_PX = 30


def main() -> int:
    server = SimServer(
        ipc_dir=REPO / "runtime" / "ipc_dumpaim",
        suite_name="libero_object",
        task_id=0,
        cam_size=256,
        observation_mode="pixels",
        pixel_grid_px=32,
    )
    server.sim.reset(init_state_idx=0)
    objects = server.sim.pack(server.sim._obs)["objects"]
    camera = Camera.from_env(server.sim.env.env, "agentview", 256, 256)
    true_u, true_v, _ = camera.project(np.asarray(objects["alphabet_soup_1"], dtype=float))
    need_u, need_v = int(round(true_u)), int(round(true_v))
    ask_u, ask_v = need_u + OFFSET_PX, need_v

    outcome = server.aim({"camera": "agent", "u": float(ask_u), "v": float(ask_v)})
    mark = outcome.measurement["mark"]
    frames = server._images_b64(mark, mark)
    for cam, data in frames.items():
        (Path("/tmp") / f"aim_{cam}.png").write_bytes(base64.b64decode(data))
    # The arm must not have moved: aim is a dry run, and this is the same assertion the
    # probe makes, checked here against the adapter's own numbers.
    print(f"status={outcome.status!r} note={outcome.note!r}")
    print(f"asked=({ask_u},{ask_v}) can_true_pixel=({need_u},{need_v})")
    print(f"eef_start={outcome.eef_start} eef_end={outcome.eef_end}")
    print(f"measurement keys={sorted(outcome.measurement)}")
    print("frames -> /tmp/aim_agent.png /tmp/aim_wrist.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
