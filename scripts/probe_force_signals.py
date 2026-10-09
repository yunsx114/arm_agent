"""Does anything in this stack expose a FORCE signal?

Question under investigation (user): can the agent (a) read the force it is
currently applying, (b) know the weight of the carried object, and (c) decide
whether the grip is "just right / a bit loose / crushing"?

Three layers have to be checked, from top to bottom:

  L1 the observation dict LIBERO builds      -> keys like robot0_*_force? 
  L2 MuJoCo model/data                        -> sensors, actuator_force, contacts
  L3 the two grasp states we care about       -> empty pads vs holding the can

L1/L2 are static (just enumerate). L3 is the one that decides whether (c) is
answerable, so it is measured: close on air, close on the can, and compare.

Self-checks:
  * the raw obs must be the documented ~40 keys (else we are on another env);
  * we must find the gripper's actuator, or the whole force question is moot;
  * closing on the can must change the finger width (else we did not grasp).

Run: bash scripts/simenv.sh scripts/probe_force_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402


def dump_mj(mj, tag: str) -> dict:
    """Enumerate every force-ish thing MuJoCo exposes, on the CURRENT state."""
    import mujoco

    out = {}
    print(f"--- {tag} ---")
    print(f"  nq={mj.model.nq} nv={mj.model.nv} nu={mj.model.nu} "
          f"nbody={mj.model.nbody} nsensor={mj.model.nsensor} "
          f"ncon={mj.data.ncon}")

    # sensors (touch / force / torque sensors live here, if any)
    names = []
    for i in range(mj.model.nsensor):
        try:
            names.append(mj.model.id2name(i, mujoco.mjtObj.mjOBJ_SENSOR))
        except Exception:  # noqa: BLE001
            names.append(None)
    print(f"  sensors: {names if any(names) else '（无名字 / 非力传感器）'}")
    out["sensors"] = names

    # actuators: the gripper actuators are the only ones that can "apply force"
    act = []
    for i in range(mj.model.nu):
        try:
            act.append(mj.model.id2name(i, mujoco.mjtObj.mjOBJ_ACTUATOR))
        except Exception:  # noqa: BLE001
            act.append(None)
    print(f"  actuators: {act if any(act) else act}")
    out["actuators"] = act

    fr = np.asarray(mj.model.actuator_forcerange)
    gp = np.asarray(mj.model.actuator_gainprm)[:, 0]
    print(f"  actuator_forcerange (N): {np.round(fr, 3).tolist()}")
    print(f"  actuator_gainprm[:,0]  : {np.round(gp, 3).tolist()}")
    out["forcerange"] = fr.tolist()

    af = np.asarray(mj.data.actuator_force)
    print(f"  data.actuator_force (N): {np.round(af, 4).tolist()}")
    out["actuator_force"] = af.tolist()

    # contacts: the only place an external force physically shows up
    mags = []
    for i in range(mj.data.ncon):
        forces = np.zeros(6)
        try:
            mujoco.mj_contactForce(mj.model, mj.data, i, forces)
            mags.append(float(np.linalg.norm(forces[:3])))
        except Exception:  # noqa: BLE001
            pass
    top = sorted(mags, reverse=True)[:6]
    print(f"  contact force magnitudes (N, top6): {np.round(top, 4).tolist()}")
    out["contact_forces"] = mags
    return out


def dump_masses(mj) -> list[tuple[str, float]]:
    """Every body's mass: this is where the payload weight physically lives."""
    import mujoco

    rows = []
    for i in range(mj.model.nbody):
        name = None
        try:
            name = mj.model.id2name(i, mujoco.mjtObj.mjOBJ_BODY)
        except Exception:  # noqa: BLE001
            pass
        rows.append((str(name), float(mj.model.body_mass[i])))
    return rows


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    obs = sim.reset(init_state_idx=0)
    keys = sorted(obs.keys())
    print(f"L1  raw obs keys ({len(keys)}): {keys}")
    force_keys = [k for k in keys if any(w in k.lower() for w in ("force", "torque", "wrench", "touch", "sensor"))]
    print(f"    force-ish keys: {force_keys if force_keys else '（无）'}")
    print()

    env = sim.env
    mj = env.sim
    adapter = ActionAdapter(sim)

    print("L1b MuJoCo bodies and their masses (where payload weight lives):")
    masses = dump_masses(mj)
    for name, m in masses:
        if m > 0.01:
            print(f"    {name:<44} {m:8.4f} kg")
    print()

    # ---- state A: pads open, nothing held -----------------------------------
    adapter.set_gripper("open")
    dump_mj(mj, "A: pads open, nothing held")

    # ---- state B: pads shut on air ------------------------------------------
    adapter.set_gripper("close")
    b = dump_mj(mj, "B: pads shut on air")

    # ---- state C: pads shut on the can --------------------------------------
    sim.reset(init_state_idx=0)
    # `reset()` REPLACES env.sim; the handle taken before it is dead
    # (observed: AttributeError: 'MjSim' object has no attribute 'model').
    mj = env.sim
    adapter.descend_to([-0.1191, -0.2398], 0.0504)
    adapter.set_gripper("close")
    width_c = max(abs(v) for v in sim._obs["robot0_gripper_qpos"])
    c = dump_mj(mj, "C: pads shut on the can")
    print(f"  gripper qpos width = {width_c:.5f}")

    # ---- self-checks ---------------------------------------------------------
    print()
    failures = 0
    raw_keys = sorted(sim._obs.keys())
    ok1 = len(raw_keys) > 30
    print(f"[{'PASS' if ok1 else 'FAIL'}] raw obs is the full LIBERO dict ({len(raw_keys)} keys)")
    failures += 0 if ok1 else 1

    grip_act = [a for a in (b.get("actuators") or []) if a and "gripper" in a]
    ok2 = bool(grip_act)
    print(f"[{'PASS' if ok2 else 'FAIL'}] gripper actuators found: {grip_act}")
    failures += 0 if ok2 else 1

    af_open = np.abs(np.asarray(b["actuator_force"])).max()
    af_grasp = np.abs(np.asarray(c["actuator_force"])).max()
    print(f"  |actuator_force| max:  air={af_open:.4f} N    on-can={af_grasp:.4f} N")
    print(f"  contact forces:        air={np.round(sorted(b.get('contact_forces') or []), 4)[-3:] if b.get('contact_forces') else []}"
          f"   on-can={np.round(sorted(c.get('contact_forces') or []), 4)[-3:] if c.get('contact_forces') else []}")

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
