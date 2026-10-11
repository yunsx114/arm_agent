import sys
from pathlib import Path
import numpy as np
REPO = Path('/root/arm_agent'); sys.path.insert(0, str(REPO/"src"))
from arm_agent.sim.action_adapter import ActionAdapter
from arm_agent.sim.libero_env import LiberoSim

sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
ad = ActionAdapter(sim)
n = int(sim.env.env.action_dim)

def show(tag):
    o = sim.pack(sim._obs)["objects"]
    c = o["alphabet_soup_1"]; b = o["basket_1"]
    eef = sim._obs["robot0_eef_pos"]
    print(f"  {tag:34s} 爪=({eef[0]:+.4f},{eef[1]:+.4f},{eef[2]:.4f}) 指宽={ad._gripper_now():.4f} "
          f"carry={ad._carrying()} | 罐=({c[0]:+.4f},{c[1]:+.4f},{c[2]:.4f}) 篮=({b[0]:+.4f},{b[1]:+.4f})")

obs = sim.reset(init_state_idx=0)
can = np.asarray(obs["objects"]["alphabet_soup_1"], dtype=float)
basket = np.asarray(obs["objects"]["basket_1"], dtype=float)
show("reset 后")
ad.descend_to((can[0], can[1]), can[2] + 0.012); show("descend_to(罐子,+1.2)")
ad.set_gripper("close");                          show("close")
ad.move("+z", 20.0);                             show("move +z 20")
ad.descend_to((basket[0], basket[1]), basket[2] + 0.25); show("descend_to(篮子,25)")
ad.set_gripper("open");                          show("open")
for _ in range(25): sim.step(np.zeros(n))
show("落地稳定后")
o = sim.pack(sim._obs)["objects"]["alphabet_soup_1"]
print(f"  篮口 x∈[{-0.0665:+.4f},{+0.0957:+.4f}] y∈[{+0.1661:+.4f},{+0.3395:+.4f}] 口沿 z=0.1513")
print(f"  罐子最终 ({o[0]:+.4f},{o[1]:+.4f},{o[2]:.4f})  success={sim.check_success()}")
sim.close()
