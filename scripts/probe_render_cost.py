"""Micro-benchmark: how much of a sim step is camera rendering?

`Observable.set_enabled(False)` should skip the per-step sensor evaluation that
renders each camera. If disabling cuts the step time drastically (expected: the
a3 project measured rendering at ~70-90% of a step), the adapter can run its
internal closed-loop ticks renderless and refresh the cameras once at the end.

    bash scripts/simenv.sh scripts/probe_render_cost.py

Also verifies that a forced refresh (`_get_observations(force_update=True)`)
restores real (non-black) frames after a renderless stretch.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.libero_env import LiberoSim

N = 30


def time_steps(sim: LiberoSim, n: int) -> float:
    t0 = time.time()
    for _ in range(n):
        sim.step(np.zeros(7))
    return (time.time() - t0) / n * 1000  # ms per step


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    sim.reset(init_state_idx=0)

    with_render = time_steps(sim, N)

    # Disable every camera-style observable.
    inner = sim.env.env  # BDDLBaseDomain (robosuite env)
    disabled = []
    for name, obs in inner._observables.items():
        if "image" in name and obs.is_enabled():
            obs.set_enabled(False)
            disabled.append(name)
    print(f"disabled observables: {disabled}")

    renderless = time_steps(sim, N)

    # Refresh: re-enable the cameras first (a disabled observable is dropped
    # from the returned dict entirely -> KeyError), then force one update.
    for name in disabled:
        inner._observables[name].set_enabled(True)
    obs = inner._get_observations(force_update=True)
    frame = np.asarray(obs["agentview_image"])
    print(f"after refresh: agentview mean={frame.mean():.1f} shape={frame.shape}")

    print(f"\nwith render : {with_render:7.1f} ms/step")
    print(f"renderless  : {renderless:7.1f} ms/step")
    print(f"render share: {(1 - renderless / with_render) * 100:5.1f}%")
    ok = renderless < with_render * 0.6 and frame.mean() > 1.0
    print("PASS" if ok else "FAIL (disable had no effect or frames are black)")
    sim.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
