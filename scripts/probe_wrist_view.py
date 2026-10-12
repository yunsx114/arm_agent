"""Why `target(camera=wrist)` cannot move the arm, and how to aim instead.

WHY THIS EXISTS (measured 2026-10-12, pixels-mode run)
 Episode 0 of the first real pixel run ended with 16 IDENTICAL calls:
 `goto_pixel(camera=wrist, u=48, v=150, offset_mm=[0,0,60])`, every one answering
 `arrived` with a measured displacement of (0,0,0) mm, and not a single `grasp` in
 80 turns. That is not the model failing to aim -- it is a property of an
 EYE-IN-HAND camera: a fixed pixel is a fixed point in the HAND's frame, so its
 back-projection moves WITH the gripper and the servo is already there. The
 grounding is degenerate, and the tool answers "arrived" forever.

 This probe measures both halves of the fix:
   1. the degeneracy itself: for the wrist camera, ground a grid of fixed pixels
      and report the world distance from the gripper to each grounded target
      (tiny and invariant = the tool can never ask for motion), and for the
      third-person camera the same numbers as a control;
   2. the AIMING SUBSTITUTE: how a world-axis nudge shows up in each image, i.e.
      "+x world moves the target N pixels in which direction", so the model can
      read the wrist view and act with `move` instead of naming pixels.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_wrist_view.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402
from arm_agent.sim.pixel_grounding import Camera, depth_metres, sample_depth  # noqa: E402

W = H = 256
NUDGE_M = 0.02


def degeneracy(env, cam_key: str, cam: Camera, eef: np.ndarray) -> None:
    print(f"\n--- {cam_key}：固定像素的反投影落点 vs 夹爪当前位置 ---")
    print(f"  {'pixel':>12s} {'depth':>7s} {'落点':28s} {'距夹爪':>8s}")
    depth_gl = depth_metres(env, cam)
    for u, v in ((48, 150), (128, 128), (128, 214), (200, 120)):
        d = sample_depth(depth_gl, u, v)
        if d is None:
            print(f"  ({u:3d},{v:3d})    深度不可用")
            continue
        p = cam.backproject(u, v, d)
        print(f"  ({u:3d},{v:3d}) {d:7.3f} {str(p.round(4)):28s} "
              f"{np.linalg.norm(p - eef) * 1000:7.1f}mm")


def axis_mapping(env, eef: np.ndarray, cam_key: str, cam: Camera) -> None:
    """How a 2 cm world nudge along +x/+y/+z shows up in this image."""
    u0, v0, _ = cam.project(eef)
    print(f"\n--- {cam_key}：世界轴微动 2cm 在画面里的表现（基准像素 ({u0:.0f},{v0:.0f})）---")
    if not cam.inside(u0, v0):
        print("  夹爪不在画面里，跳过")
        return
    print(f"  {'世界方向':>10s} {'Δu(像素)':>10s} {'Δv(像素)':>10s}   解读")
    for label, vec in (
        ("+x", np.array([1.0, 0, 0])),
        ("-x", np.array([-1.0, 0, 0])),
        ("+y", np.array([0, 1.0, 0])),
        ("-y", np.array([0, -1.0, 0])),
        ("+z (抬高)", np.array([0, 0, 1.0])),
        ("-z (下降)", np.array([0, 0, -1.0])),
    ):
        u, v, _ = cam.project(eef + vec * NUDGE_M)
        du, dv = u - u0, v - v0
        if abs(du) < 0.01 and abs(dv) < 0.01:
            read = "几乎不变（垂直于画面平面）"
        else:
            read = ""
            if abs(du) >= 1:
                read += f"向右 {du:+.0f}" if du > 0 else f"向左 {du:+.0f}"
            if abs(dv) >= 1:
                read += ("，" if read else "") + (f"向下 {dv:+.0f}" if dv > 0 else f"向上 {dv:+.0f}")
            read += " 像素"
        print(f"  {label:>10s} {du:10.1f} {dv:10.1f}   {read}")


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=W)
    adapter = ActionAdapter(sim)
    env = sim.env.env
    obs = sim.reset(init_state_idx=0)
    can = np.asarray(obs["objects"]["alphabet_soup_1"], dtype=float)

    print("=" * 78)
    print("A. 腕部相机（装在夹爪上）")
    print("=" * 78)
    agent = Camera.from_env(env, "agentview", W, H)
    wrist = Camera.from_env(env, "robot0_eye_in_hand", W, H)
    eef_home = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
    degeneracy(env, "wrist(home)", wrist, eef_home)

    # 移动到抓取高度（腕部画面里能看到罐子的姿态）
    adapter.release()
    adapter.descend_to((can[0], can[1]), can[2] + 0.12)
    eef_mid = np.asarray(sim._obs["robot0_eef_pos"], dtype=float)
    wrist = Camera.from_env(env, "robot0_eye_in_hand", W, H)
    agent = Camera.from_env(env, "agentview", W, H)
    print(f"\n[setup] 已降到罐子上方 12cm：eef={eef_mid.round(4)}")
    degeneracy(env, "wrist(12cm)", wrist, eef_mid)
    axis_mapping(env, eef_mid, "wrist(12cm)", wrist)

    print("\n" + "=" * 78)
    print("B. 第三人称（对照组）")
    print("=" * 78)
    degeneracy(env, "agentview", agent, eef_mid)
    axis_mapping(env, eef_mid, "agentview", agent)

    # 罐子在腕部画面里的位置与大小（模型实际要瞄的东西）
    u, v, d = wrist.project(can)
    print(f"\n[参考] 罐心投影到腕部画面 = ({u:.0f},{v:.0f})，深度 {d:.3f}m，"
          f"{wrist.mm_per_px(d):.2f}mm/像素（画面中心 (128,128)）")
    hu, hv, _ = wrist.project(can + np.array([0.0166, 0, 0]))
    print(f"       罐子半径 16.6mm ⇒ 在画面里约 {abs(hu - u):.1f} 像素（直径 ~{abs(hu - u) * 2:.0f}px）")

    sim.close()
    return 0


raise SystemExit(main())
