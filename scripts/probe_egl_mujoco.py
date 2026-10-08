"""Phase 0 · E2 minimal probe: can this machine render MuJoCo offscreen via software EGL?

Pure mujoco (no robosuite/LIBERO), so it isolates the render stack.

Env (from a3_dual_arm_sim's verified setup):
    GL_SW=/lab/haoq_lab/cse12311731/miniconda3/envs/gl_sw
    LD_LIBRARY_PATH=$GL_SW/lib LIBGL_DRIVERS_PATH=$GL_SW/lib/dri MUJOCO_GL=egl

Prints: render resolution, mean pixel value (all-black = EGL silently broken),
per-render milliseconds (the budget driver for the episode loop).
"""

from __future__ import annotations

import os
import time

import numpy as np

XML = """
<mujoco>
  <visual><global offwidth="256" offheight="256"/></visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0.35 0.1 0.06">
      <geom type="box" size="0.05 0.05 0.05" rgba="0.9 0.1 0.1 1"/>
    </body>
    <camera name="view" pos="0.9 -0.9 0.9" xyaxes="1 1 0 -0.5 0.5 1"/>
  </worldbody>
</mujoco>
"""


def main() -> int:
    print("[e2] MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    print("[e2] LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
    print("[e2] LIBGL_DRIVERS_PATH =", os.environ.get("LIBGL_DRIVERS_PATH"))

    import mujoco

    print("[e2] mujoco", mujoco.__version__)
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, 256, 256)
    t0 = time.time()
    n = 20
    for _ in range(n):
        renderer.update_scene(data, camera="view")
        img = renderer.render()
    per = (time.time() - t0) / n
    img = np.asarray(img)
    print(f"[e2] image {img.shape} dtype={img.dtype} mean={img.mean():.2f} std={img.std():.2f}")
    print(f"[e2] unique colors: {len(np.unique(img.reshape(-1, 3), axis=0))}")
    print(f"[e2] per-render: {per * 1000:.0f} ms")
    if img.mean() < 1e-6:
        print("[e2] FAIL: image is black - EGL context is dead")
        return 1
    print("[e2] PASS: software EGL rendering works")
    # mujoco 2.3.7's Renderer has no close(); its GLContext destructor may print
    # an EGLError at interpreter exit (benign teardown noise on this stack).
    if hasattr(renderer, "close"):
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
