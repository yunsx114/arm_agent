"""What does the raw contact table look like at the moment of a grasp?

WHY: the grasp concept (档 A) must be decided from CONTACT, not from the finger
width, and it must not know any object name.  That means reading
`sim.data.contact` and mapping geom -> body -> name.  Every name below is
therefore measured here rather than assumed: three consecutive bugs in this
project were "a value from a different vocabulary" (see probe_tool_wiring.py),
and a wrong body-name pattern would be a fourth.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_contact_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

CONTACT_TOL_M = 0.0005


def raw_model(env):
    """The native MjModel (robosuite wraps it; `._model` is the real one)."""
    return getattr(env.sim.model, "_model", env.sim.model)


def names(env) -> list[str]:
    """Body names, MEASURED: `sim.model.body_id2name` does NOT exist on the
    robosuite wrapper (the first version of this probe printed `<body N>` for
    every body and therefore found no finger at all). The wrapper exposes the
    whole list as `body_names`; that is what probe_rim.py used too."""
    return list(getattr(env.sim.model, "body_names", []) or [])


def body_name(env, body_id: int) -> str:
    table = names(env)
    if 0 <= int(body_id) < len(table):
        return str(table[int(body_id)])
    return f"<body {body_id}>"


def contact_pairs(env, tol: float = CONTACT_TOL_M) -> list[tuple[str, str, float]]:
    m, d = raw_model(env), env.sim.data
    out = []
    for c in d.contact[: int(d.ncon)]:
        if float(c.dist) > tol:
            continue
        b1 = int(m.geom_bodyid[int(c.geom1)])
        b2 = int(m.geom_bodyid[int(c.geom2)])
        out.append((body_name(env, b1), body_name(env, b2), float(c.dist)))
    return out


def show(label: str, env, adapter: ActionAdapter) -> None:
    width = adapter._gripper_now()
    pairs = contact_pairs(env)
    table = names(env)
    print(f"\n=== {label} ===")
    print(f"  指宽 {width:.4f}  ncon={int(env.sim.data.ncon)}  接触对 {len(pairs)}")
    print(f"  手指类 body: {[n for n in table if 'finger' in n.lower() or 'gripper' in n.lower()]}")
    touched = {b for pair in pairs for b in pair[:2]}
    print(f"  接触中出现的 body: {sorted(touched)}")
    fingers = [n for n in table if "finger" in n.lower()]
    interesting = [p for p in pairs if p[0] in fingers or p[1] in fingers]
    print(f"  含手指的接触对: {len(interesting)}")
    for a, b, dist in interesting[:14]:
        print(f"    {a:28s} <-> {b:28s}  dist={dist * 1000:+.2f} mm")


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    env = sim.env.env
    adapter = ActionAdapter(sim)
    n_act = int(env.action_dim)

    obs = sim.reset(init_state_idx=0)
    print("=== 初始 ===")
    print(f"  指宽 {adapter._gripper_now():.4f}（仿真启动就是半闭的）")
    show("初始（未动）", env, adapter)

    can = np.asarray(obs["objects"]["alphabet_soup_1"], dtype=float)
    adapter.set_gripper("open")
    show("open 之后", env, adapter)
    adapter.descend_to((can[0], can[1]), can[2] + 0.012)
    show("下降对准到抓取高度（两指间有罐子，但还没夹）", env, adapter)
    adapter.set_gripper("close")
    show("close 之后（应当是双向接触）", env, adapter)

    adapter.move("+z", 15.0)
    show("抬起 15cm 之后（应当仍双向接触）", env, adapter)

    # 空夹对照：抬高后张开再闭合
    adapter.set_gripper("open")
    adapter.set_gripper("close")
    show("空中空夹（应当无接触）", env, adapter)

    for _ in range(5):
        sim.step(np.zeros(n_act))
    sim.close()
    return 0


raise SystemExit(main())
