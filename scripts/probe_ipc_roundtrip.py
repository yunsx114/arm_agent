"""M1 IPC round-trip probe: exercise the NFS protocol end to end on the login node.

Runs the server IN-PROCESS (thread) and drives it with the real client, so it
validates the on-disk protocol without needing the actual two-node setup:

    cd arm_agent && bash scripts/simenv.sh scripts/probe_ipc_roundtrip.py

Checks: ping -> reset -> state -> move -> look -> success -> act(gripper) ->
close, measuring the round-trip time of each call (budget: ms-level file ops +
the server's own work; anything else is a protocol bug).
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.client import SimClient
from arm_agent.sim.server import SimServer


def main() -> int:
    ipc_dir = REPO / "runtime" / "ipc_probe"
    shutil.rmtree(ipc_dir, ignore_errors=True)

    server = SimServer(ipc_dir=ipc_dir, suite_name="libero_object", task_id=0, cam_size=256)
    thread = threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 120}, daemon=True)
    thread.start()
    time.sleep(0.2)

    client = SimClient(ipc_dir)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    t0 = time.time()
    resp = client.ping()
    check("ping", resp.result.get("pong") is True, f"{time.time() - t0:.3f}s task={resp.result.get('task')}")

    t0 = time.time()
    resp = client.reset(init_state_idx=0)
    obs = resp.obs
    images = resp.images(ipc_dir)
    check(
        "reset",
        obs.get("step_count") == 0 and set(images) == {"agent", "wrist"},
        f"{time.time() - t0:.2f}s eef={obs.get('eef_pos')} images={sorted(images)}",
    )

    t0 = time.time()
    resp = client.state()
    check("state", "eef_pos" in resp.obs, f"{time.time() - t0:.3f}s")

    t0 = time.time()
    resp = client.act({"kind": "move", "direction": "+z", "distance_cm": 3.0})
    out = resp.outcome
    moved = out.get("measured_delta_m", [0, 0, 0])[2]
    check(
        "act move +z 3cm",
        abs(moved - 0.03) < 0.012,
        f"{time.time() - t0:.2f}s steps={out.get('steps')} dz={moved:+.4f}",
    )

    t0 = time.time()
    resp = client.look()
    check("look", "images" in resp.result, f"{time.time() - t0:.2f}s")

    t0 = time.time()
    ok = client.success()
    check("success query", ok is False, f"{time.time() - t0:.3f}s success={ok}")

    t0 = time.time()
    resp = client.act({"kind": "gripper", "action": "close"})
    check(
        "act gripper close",
        "closed" in resp.outcome.get("note", ""),
        f"{time.time() - t0:.2f}s note={resp.outcome.get('note')}",
    )

    client.close()
    time.sleep(0.5)

    leftovers = sorted(p.name for p in ipc_dir.glob("*.json"))
    check("no leftover json (protocol cleaned up)", not leftovers, f"leftovers={leftovers}")

    print(f"\n=== {len(failures)} failure(s): {failures} ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
