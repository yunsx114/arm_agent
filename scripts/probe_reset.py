import sys
from pathlib import Path
import numpy as np
REPO = Path('/root/arm_agent'); sys.path.insert(0, str(REPO / "src"))
from arm_agent.sim.libero_env import LiberoSim
sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
print("连续 reset 5 次，看物体位置是否每次都一样：")
for i in range(5):
    obs = sim.reset(init_state_idx=0)
    o = obs["objects"]
    print(f"  #{i}: can=({o['alphabet_soup_1'][0]:+.4f},{o['alphabet_soup_1'][1]:+.4f},{o['alphabet_soup_1'][2]:+.4f})"
          f"  basket=({o['basket_1'][0]:+.4f},{o['basket_1'][1]:+.4f},{o['basket_1'][2]:+.4f})"
          f"  success={sim.check_success()}")
# 再测：动一下再 reset
print("动一步 (关爪 + 横移) 之后 reset：")
obs = sim.reset(init_state_idx=0)
for _ in range(40): sim.step(np.array([0,0,0,0,0,0,1.0]))     # 关爪
for _ in range(40): sim.step(np.array([0,0,0,0.5,0,0,0]))     # 平移
print(f"  动完后 can=({sim.pack(sim._obs)['objects']['alphabet_soup_1']})")
obs = sim.reset(init_state_idx=0)
o = obs["objects"]
print(f"  reset 后 can=({o['alphabet_soup_1'][0]:+.4f},{o['alphabet_soup_1'][1]:+.4f},{o['alphabet_soup_1'][2]:+.4f})")
sim.close()
