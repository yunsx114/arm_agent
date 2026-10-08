"""Phase 0 · E1/E2 smoke test: load a LIBERO task, render, step, and time it.

Run inside the `libero` conda env with software EGL on the login node:

    GL_SW=/lab/haoq_lab/cse12311731/miniconda3/envs/gl_sw
    LD_LIBRARY_PATH=$GL_SW/lib LIBGL_DRIVERS_PATH=$GL_SW/lib/dri MUJOCO_GL=egl \
    /lab/haoq_lab/cse12311731/miniconda3/envs/libero/bin/python smoke_libero_e1.py

Checks (in order):
 1. benchmark registry loads, task metadata readable
 2. env constructs + reset + set_init_state (official demo flow: 5 zero steps first)
 3. camera observations present, images saved for visual inspection
 4. wall-clock per step (render vs physics split not separable from outside,
    but repeated steps give the average)
 5. check_success() callable; state snapshot save/restore works
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "smoke"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--cam", type=int, default=256, help="camera size")
    parser.add_argument("--steps", type=int, default=10, help="timed dummy steps")
    args = parser.parse_args()

    print("[smoke] MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    world = time.time()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    print(f"[smoke] import libero OK ({time.time() - world:.1f}s)")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    print(f"[smoke] task {args.task_id}: {task.name}")
    print(f"[smoke] language: {task.language}")
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    assert os.path.exists(bddl), f"bddl not found: {bddl}"

    t0 = time.time()
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=args.cam,
        camera_widths=args.cam,
    )
    print(f"[smoke] env constructed in {time.time() - t0:.1f}s")

    t0 = time.time()
    env.seed(0)
    env.reset()
    print(f"[smoke] reset in {time.time() - t0:.1f}s")

    init_states = suite.get_task_init_states(args.task_id)
    obs = env.set_init_state(init_states[0])
    print(f"[smoke] init_states shape: {np.asarray(init_states).shape}")

    # Official demo convention: 5 zero-action steps to settle + keep gripper open.
    for _ in range(5):
        obs, reward, done, info = env.step(np.zeros(7))

    # --- obs inspection ---
    keys = sorted(obs.keys())
    print(f"[smoke] obs keys ({len(keys)}): {keys}")
    av = np.asarray(obs["agentview_image"])
    wr = np.asarray(obs["robot0_eye_in_hand_image"])
    print(f"[smoke] agentview {av.shape} dtype={av.dtype} mean={av.mean():.1f}")
    print(f"[smoke] wrist     {wr.shape} dtype={wr.dtype} mean={wr.mean():.1f}")
    print(f"[smoke] eef_pos: {np.asarray(obs['robot0_eef_pos']).round(4)}")
    print(f"[smoke] eef_quat: {np.asarray(obs['robot0_eef_quat']).round(4)}")
    print(f"[smoke] gripper_qpos: {np.asarray(obs['robot0_gripper_qpos']).round(4)}")
    if "object-state" in obs:
        print(f"[smoke] object-state dim: {np.asarray(obs['object-state']).shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        # LIBERO images are vertically flipped for display.
        Image.fromarray(av[::-1]).save(OUT_DIR / f"e1_{args.suite}_{args.task_id}_agentview.png")
        Image.fromarray(wr[::-1]).save(OUT_DIR / f"e1_{args.suite}_{args.task_id}_wrist.png")
        print(f"[smoke] images saved to {OUT_DIR}")
    except ImportError:
        print("[smoke] PIL missing, skip image save")

    # --- timing ---
    ts = []
    for _ in range(args.steps):
        t0 = time.time()
        obs, reward, done, info = env.step(np.zeros(7))
        ts.append(time.time() - t0)
    ts = np.asarray(ts)
    print(f"[smoke] step time: mean={ts.mean() * 1000:.0f}ms min={ts.min() * 1000:.0f}ms max={ts.max() * 1000:.0f}ms")

    # --- success check + state snapshot round trip ---
    print(f"[smoke] check_success(): {env.check_success()}")
    try:
        state = env.get_sim_state()
        env.set_init_state(state)
        print("[smoke] sim state save/restore OK")
    except Exception as exc:  # noqa: BLE001 - exploratory probe
        print(f"[smoke] sim state snapshot FAILED: {exc}")

    env.close()
    print("[smoke] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
