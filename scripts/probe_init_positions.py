#!/usr/bin/env python3
"""Print the payload/basket ground truth for a range of init states.

WHY THIS EXISTS: the 档 D question is "did the ladder get the gripper closer to the can",
and both arms of that comparison (raw baseline, ladder run) have to be measured the SAME
way. The transcript carries the END EFFECTOR's own positions (`measurement.eef_end`) but
never the objects' -- that is the whole point of `raw` mode -- so the truth comes from the
simulator here, once per init state, and the per-episode minimum is then computed offline
by `audit_ladder_distance.py`.

Objects do not move between the reset and the episode's own actions (they can only be
pushed BY the arm, which is what we are measuring), so a reset alone is enough.

USAGE
    bash scripts/simenv.sh scripts/probe_init_positions.py [start] [count]
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.client import SimClient  # noqa: E402
from arm_agent.sim.server import SimServer  # noqa: E402


def main() -> int:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ipc_dir = REPO / "runtime" / "ipc_truth"
    shutil.rmtree(ipc_dir, ignore_errors=True)
    server = SimServer(ipc_dir=ipc_dir, suite_name="libero_object", task_id=0, cam_size=256)
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 120}, daemon=True)
    thread.start()
    time.sleep(0.2)
    client = SimClient(ipc_dir)

    out = {}
    for idx in range(start, start + count):
        resp = client.reset(init_state_idx=idx)
        objects = resp.obs.get("objects", {})
        can = objects.get("alphabet_soup_1")
        basket = objects.get("basket_1")
        out[str(idx)] = {"can": [round(float(v), 4) for v in (can or [])],
                         "basket": [round(float(v), 4) for v in (basket or [])]}
        print(f"init_state {idx}: can={out[str(idx)]['can']} basket={out[str(idx)]['basket']}")
    (REPO / "outputs" / "init_truth.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"written: {REPO / 'outputs' / 'init_truth.json'}")
    client.close()
    time.sleep(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
