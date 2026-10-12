#!/usr/bin/env python3
"""Is the axis convention in `skills/scene_tools.md` actually TRUE in the image?

WHY: measured over 4 arms (24 episodes), the model's `+y` choices agreed with the real
direction to the can **0-2% of the time** (`-y` was 94-100%), and `+x` only 13-29%. Two
explanations fit that number and they need completely different fixes:

  (a) the skill's convention is BACKWARDS, so a model that reads the image correctly is
      told to move the wrong way -- a one-line fix worth everything; or
  (b) the convention is right and the model is pointing at the wrong OBJECT (the object
      row and the basket sit on the +y side; the can is alone in the -y quadrant).

This probe settles it from the simulator instead of from argument: it projects the gripper
and the can into the agent view, then physically walks +y and +x and re-projects. Whatever
the crosshair does in the image IS the convention, and the can's side of the crosshair IS
the truth the model is being asked to report.

USAGE
    bash scripts/simenv.sh scripts/probe_axis_convention.py
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arm_agent import contracts as C  # noqa: E402
from arm_agent.sim.client import SimClient  # noqa: E402
from arm_agent.sim.pixel_grounding import Camera  # noqa: E402
from arm_agent.sim.server import SimServer  # noqa: E402


def project(server: SimServer, key: str, point) -> tuple[float, float]:
    cam = Camera.from_env(server.sim.env.env, C.CAMERA_NAMES[key], 256, 256)
    u, v, _ = cam.project(np.asarray(point, dtype=float))
    return float(u), float(v)


def main() -> int:
    ipc_dir = REPO / "runtime" / "ipc_axis"
    shutil.rmtree(ipc_dir, ignore_errors=True)
    server = SimServer(ipc_dir=ipc_dir, suite_name="libero_object", task_id=0, cam_size=256)
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 120}, daemon=True)
    thread.start()
    time.sleep(0.2)
    client = SimClient(ipc_dir)

    obs = client.reset(init_state_idx=0).obs
    objects = obs.get("objects", {})
    # The packed obs names it `eef_pos`; the raw obs uses the robosuite key. Accept both.
    eef_raw = obs.get("eef_pos", obs.get(C.OBS_EEF_POS))
    can = np.asarray(objects["alphabet_soup_1"], dtype=float)
    print(f"reset 时的爪心（来自打包观测）= {[round(float(v), 4) for v in (eef_raw or [])]}")

    def show(tag: str) -> None:
        e = np.asarray(server.sim._obs[C.OBS_EEF_POS], dtype=float)
        ue, ve = project(server, "agent", e)
        uc, vc = project(server, "agent", can)
        print(f"{tag:>10} 爪心世界=({e[0]:+.3f},{e[1]:+.3f},{e[2]:+.3f})  "
              f"像素=({ue:6.1f},{ve:6.1f})   罐心像素=({uc:6.1f},{vc:6.1f})   "
              f"罐在爪心 {'右' if uc > ue else '左'}{abs(uc - ue):5.1f}px / "
              f"{'下' if vc > ve else '上'}{abs(vc - ve):5.1f}px")

    print(f"罐心世界坐标 = ({can[0]:+.4f},{can[1]:+.4f},{can[2]:+.4f})")
    print("（图像坐标：u 向右增大，v 向下增大）")
    show("reset")
    for direction, label in (("+y", "加 y"), ("+x", "加 x"), ("+z", "加 z")):
        client.act({"kind": "move", "direction": direction, "distance_cm": 5.0})
        show(f"{direction} 5cm")

    print()
    print("读数要点：")
    print("  * 上面每一步 '罐在爪心的左右上下' 就是 skill 里那句约定的实测版本。")
    print("  * 如果 +y 让像素 u 变小（向左），而罐子在爪心的右边，那么'往 +y 走以靠近右边的罐子'")
    print("    就是错的——模型没错，是约定反了。")
    client.close()
    time.sleep(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
