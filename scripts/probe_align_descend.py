"""M1 align/descend end-to-end probe (regression guard).

Two bugs were found the hard way and must never come back silently:

  1. `ActionAdapter.move()` never called `sim.step()` inside its closed loop,
     so every `move` returned 0 ticks / 0 displacement. `descend_to` delegates
     each step to `move`, so the ENTIRE grasping path was dead while the run
     still reported "success: False, retries: 0" (no error anywhere).
  2. `SimServer` read `objects` off `self.sim._obs`, but "objects" is
     SYNTHESISED by `pack()` from the raw `<name>_pos` keys. The lookup always
     returned {}, so `align`/`descend` always raised
     "no object matching ...; have []".

Runs the server IN-PROCESS (thread) and drives it with the real client, so it
exercises the exact code path the LLM uses:

    cd arm_agent && bash scripts/simenv.sh scripts/probe_align_descend.py

Asserts on MEASURED geometry (never on formulas):
  align   -> eef xy within 5 mm of the object xy
  descend -> eef xy within 5 mm AND eef z within 6 mm of obj_z + 1.2 cm
  close   -> finger width ~0.02 (an object between the pads, not air ~0.0012)
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
    ipc_dir = REPO / "runtime" / "ipc_probe_align"
    shutil.rmtree(ipc_dir, ignore_errors=True)

    server = SimServer(ipc_dir=ipc_dir, suite_name="libero_object", task_id=0, cam_size=256)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"idle_exit_s": 180}, daemon=True
    )
    thread.start()
    time.sleep(0.2)

    client = SimClient(ipc_dir)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    resp = client.reset(init_state_idx=0)
    obs = resp.obs
    objs: dict[str, list[float]] = obs["objects"]
    print(f"objects available: {sorted(objs)}")

    cand = [k for k in objs if "soup" in k]
    target = cand[0] if cand else sorted(objs)[0]
    pos = objs[target]
    print(f"target={target} pos={pos}  (this is what the model asks for as 'alphabet_soup')")

    # ---------------------------------------------------------------- align
    t0 = time.time()
    resp = client.act({"kind": "align", "name": "alphabet_soup"})
    out = resp.outcome
    end = out["eef_end"]
    xy_err = ((end[0] - pos[0]) ** 2 + (end[1] - pos[1]) ** 2) ** 0.5
    check(
        "align_xy reaches the object's xy",
        xy_err < 0.005,
        f"{time.time() - t0:.1f}s xy_err={xy_err * 1000:.1f}mm "
        f"steps={out['steps']} note={out['note']!r}",
    )

    # -------------------------------------------------------------- descend
    t0 = time.time()
    resp = client.act({"kind": "descend", "name": "alphabet_soup", "z_offset_cm": 1.2})
    out = resp.outcome
    end = out["eef_end"]
    xy_err = ((end[0] - pos[0]) ** 2 + (end[1] - pos[1]) ** 2) ** 0.5
    z_err = abs(end[2] - (pos[2] + 0.012))
    check(
        "descend_to reaches (obj xy, obj_z + 1.2cm)",
        xy_err < 0.005 and z_err < 0.006,
        f"{time.time() - t0:.1f}s xy_err={xy_err * 1000:.1f}mm "
        f"z_err={z_err * 1000:.1f}mm steps={out['steps']} note={out['note']!r}",
    )

    # ---------------------------------------------------------------- grasp
    t0 = time.time()
    resp = client.act({"kind": "gripper", "action": "close"})
    out = resp.outcome
    width = float(out["note"].split("width ")[1].split()[0])
    check(
        "close grasps the can (width ~0.02, not air ~0.0012)",
        width > 0.01,
        f"{time.time() - t0:.1f}s width={width:.4f}",
    )

    # ----------------------------------------------------------------- lift
    resp = client.act({"kind": "move", "direction": "+z", "distance_cm": 5})
    dz = resp.outcome["measured_delta_m"][2]
    check("lift +z 5cm actually moves", dz > 0.035, f"dz={dz:+.4f}")
    lifted_pos = resp.obs["objects"][target][2]
    print(f"(can z after lift: {lifted_pos:.4f}; started {pos[2]:.4f})")

    print(f"\n=== {len(failures)} failure(s): {failures} ===")
    try:
        client.close()
    except Exception:  # noqa: BLE001 - probe teardown only
        pass
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
