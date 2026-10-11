"""What exactly does the sim accept as "placed in the basket"?

WHY THIS EXISTS
  The goal predicate of `pick_up_the_alphabet_soup_and_place_it_in_the_basket`
  is `(In alphabet_soup_1 basket_1_contain_region)`.  In LIBERO that expands to

      In(a, b) = b.check_contact(a) and b.check_contain(a)

  and when `b` is a *site* (``*_contain_region``) `check_contact` returns True
  unconditionally (sites have no dynamics), so the whole verdict collapses to

      SiteObject.in_box(site_pos, site_mat, soup_body_pos)
        total_size = |site_mat @ site_size|
        lb = site_pos - total_size;  lb[2] -= 0.01
        ub = site_pos + total_size
        True  <=>  all(lb < soup_origin < ub)

  i.e. THE CAN'S BODY ORIGIN (= its centre) must fall inside one axis-aligned
  box derived from the site.  Every "the can looks like it went in but the sim
  said no" incident is a claim about this box, so measure the box once instead
  of arguing about it.

WHAT IT DOES
  1. prints the site's raw pos/size/mat and the resulting lb/ub;
  2. runs one clean release at a given offset and reports where the can settles
     and whether the box accepts it;
  3. teleports the can straight up in 1 cm steps through the box and prints
     `in_box` at every height -- the acceptance band, read off the instrument
     rather than inferred.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_accept_box.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

CAN_HALF_HEIGHT = 0.0426  # can body origin (= centre) sits this far above its bottom
SITE_NAME = "basket_1_contain_region"


def box_of(env, site_name: str):
    """Replicate SiteObjectState.check_contain / SiteObject.in_box."""
    site = env.object_sites_dict[site_name]
    pos = np.asarray(env.sim.data.get_site_xpos(site_name), dtype=float)
    mat = np.asarray(env.sim.data.get_site_xmat(site_name), dtype=float)
    size = np.asarray(site.size, dtype=float).reshape(3)
    total = np.abs(mat @ size)
    lb = pos - total
    lb[2] -= 0.01
    return site, pos, mat, size, lb, pos + total


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    env = sim.env.env
    adapter = ActionAdapter(sim)
    n_act = int(env.action_dim)

    obs = sim.reset(init_state_idx=0)
    print("=== 所有 contain region site ===")
    for k in sorted(env.object_sites_dict):
        if "contain" in k or "region" in k:
            print(f"  {k}")

    site, spos, smat, ssize, lb, ub = box_of(env, SITE_NAME)
    print(f"\n=== 接受盒 {SITE_NAME} ===")
    print(f"  site pos (世界) = ({spos[0]:+.4f}, {spos[1]:+.4f}, {spos[2]:+.4f})")
    print(f"  site size       = {ssize}   (type={site.site_type})")
    print(f"  lb              = ({lb[0]:+.4f}, {lb[1]:+.4f}, {lb[2]:+.4f})")
    print(f"  ub              = ({ub[0]:+.4f}, {ub[1]:+.4f}, {ub[2]:+.4f})")
    basket = np.asarray(obs["objects"]["basket_1"], dtype=float)
    print(f"  篮子 body       = ({basket[0]:+.4f}, {basket[1]:+.4f}, {basket[2]:+.4f})")
    print(f"  site 相对篮子   = ({spos[0]-basket[0]:+.4f}, {spos[1]-basket[1]:+.4f}, "
          f"{spos[2]-basket[2]:+.4f})")
    soup0 = np.asarray(obs["objects"]["alphabet_soup_1"], dtype=float)
    print(f"  罐子初始 body   = ({soup0[0]:+.4f}, {soup0[1]:+.4f}, {soup0[2]:+.4f})")
    print(f"  罐子初始 in_box = {bool(np.all(soup0 > lb) and np.all(soup0 < ub))}")

    # ------------------------------------------------ one clean release, offset 25
    print("\n=== 干净放手（carry 25cm，offset 25）后罐子落点 ===")
    obs = sim.reset(init_state_idx=0)
    can = np.asarray(obs["objects"]["alphabet_soup_1"], dtype=float)
    basket = np.asarray(obs["objects"]["basket_1"], dtype=float)
    adapter.set_gripper("open")
    adapter.descend_to((can[0], can[1]), can[2] + 0.012)
    adapter.set_gripper("close")
    carried = adapter._carrying()
    adapter.move("+z", 25.0)
    adapter.descend_to((basket[0], basket[1]), basket[2] + 0.25)
    eef = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
    adapter.set_gripper("open")
    for _ in range(40):
        sim.step(np.zeros(n_act))
    after = sim.pack(sim._obs)["objects"]
    ce = np.asarray(after["alphabet_soup_1"], dtype=float)
    be = np.asarray(after["basket_1"], dtype=float)
    _, spos2, _, _, lb2, ub2 = box_of(env, SITE_NAME)
    inside = bool(np.all(ce > lb2) and np.all(ce < ub2))
    print(f"  carried={carried}  eef_z={eef[2]:.4f}  eef 松手时物体底={eef[2]-0.05:.4f}")
    print(f"  落点 body = ({ce[0]:+.4f}, {ce[1]:+.4f}, {ce[2]:+.4f})  底={ce[2]-CAN_HALF_HEIGHT:.4f}")
    print(f"  篮子 body = ({be[0]:+.4f}, {be[1]:+.4f}, {be[2]:+.4f})")
    print(f"  in_box={inside}  success={bool(sim.check_success())}")
    print(f"  越界量: 上={np.round(ce - ub2, 4)} 下={np.round(lb2 - ce, 4)}")

    # ------------------------------------------------ read the acceptance band
    print("\n=== 把罐子沿 z 平移，看接受带（xyz 保持在上面的落点 xy）===")
    obj = env.get_object("alphabet_soup_1")
    addr = env.sim.model.get_joint_qpos_addr(obj.joints[0])
    if isinstance(addr, (tuple, list)):
        addr = int(addr[0])
    base = np.array(env.sim.data.qpos[addr : addr + 7], dtype=float)
    for z in np.arange(0.00, 0.26, 0.01):
        env.sim.data.qpos[addr : addr + 7] = [base[0], base[1], float(z), 1, 0, 0, 0]
        env.sim.forward()
        pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id["alphabet_soup_1"]], dtype=float)
        _, sp, _, _, l, u = box_of(env, SITE_NAME)
        ok = bool(np.all(pos > l) and np.all(pos < u))
        print(f"  z={z:.2f} -> body_z={pos[2]:.4f} in_box={ok} success={bool(sim.check_success())}")

    sim.close()
    return 0


raise SystemExit(main())
