"""Rim height + the task's real success predicate (robosuite-safe).

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_rim.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.libero_env import LiberoSim  # noqa: E402


def show(label, fn) -> None:
    print(f"\n=== {label} ===")
    try:
        print(inspect.getsource(fn))
    except Exception as exc:
        print(f"  (取源码失败: {type(exc).__name__}: {exc})")


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    env = sim.env.env
    obs = sim.reset(init_state_idx=0)
    print("=== 物体真值 ===")
    for k, v in sorted(obs["objects"].items()):
        print(f"  {k:22s} ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")

    print("\n=== 篮子/罐子 body 的几何 z 范围 ===")
    try:
        m, d = env.sim.model, env.sim.data
        names = list(getattr(m, "body_names", []) or [])
        body_of_geom = np.asarray(m.geom_bodyid, dtype=int)
        gsize = np.asarray(m.geom_size); gpos = np.asarray(d.geom_xpos); gmat = np.asarray(d.geom_xmat)
        for b, name in enumerate(names):
            if not any(t in (name or "").lower() for t in ("basket", "soup", "tray", "container", "bin", "can")):
                continue
            lo, hi = float("inf"), float("-inf")
            for g in np.nonzero(body_of_geom == b)[0]:
                mat = gmat[g].reshape(3, 3)
                corners = np.array([[sx*gsize[g][0], sy*gsize[g][1], sz*gsize[g][2]]
                                    for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
                zs = (corners @ mat.T + gpos[g])[:, 2]
                lo, hi = min(lo, float(zs.min())), max(hi, float(zs.max()))
            if np.isfinite(lo):
                print(f"  {name:26s} z ∈ [{lo:+.4f}, {hi:+.4f}]  高 {hi-lo:.4f} m")
    except Exception as exc:
        print(f"  (几何读取失败: {type(exc).__name__}: {exc})")

    show("check_success", sim.env.check_success)
    inner = getattr(env, "_check_success", None)
    if inner is not None:
        show("robosuite env._check_success", inner)
    for attr in ("_goal", "goal", "_task", "task", "reward_shaping"):
        obj = getattr(env, attr, None)
        if obj is not None and not callable(obj):
            print(f"\n=== {attr} = {obj!r} ===")
    sim.close()
    return 0


raise SystemExit(main())
