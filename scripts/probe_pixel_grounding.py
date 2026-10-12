"""How accurate is "pick a pixel, get a world point"? (档 B, prerequisite)

WHY THIS EXISTS
 档 B replaces `approach(<object name>)` -- which reads the sim's ground-truth
 object pose -- with the model choosing a PIXEL and the code back-projecting it
 through depth + camera calibration (the pattern in FBddcz/embodied-jev
 `libero_policy.resolve_plan`, target.kind="pixel"). Before writing that tool we
 need the error budget, because the numbers decide the design:

   * pixel granularity sets how precisely a target can be named at all, and this
     project already measured that grasping needs ~2 mm of alignment;
   * that decides whether `goto_pixel` can be one open-loop hop or must be
     two-stage (coarse from the third-person view, then a close-range view or an
     image-space servo), and how fine a coordinate grid the model must be shown.

WHAT IT MEASURES (and what it deliberately does NOT)
 The recovered point of a pixel lands on the visible SURFACE, so its distance to
 an object's CENTRE is the object's near half-thickness, not an error. This probe
 therefore separates:
   A. correctness of the maths -- table-plane coplanarity over a 16x16 pixel grid
      (a flat surface must come back flat), plus a project->backproject->project
      round trip;
   B. orientation of the depth buffer (row 0 = top or bottom) -- decided by
      OBJECT FIDELITY against ground truth, because the wrong flip reads the
      depth of a different surface;
   C. the surface-vs-centre offset per object (this is the clearance offset the
      model has to supply) and the granularity cost (mm/px, +-1/2 px, grid
      snapping).

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_pixel_grounding.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

W = H = 256
GRID_PITCHES = (4, 8, 16, 32, 64)
EEF_KEY = "robot0_eef_pos"


class Camera:
    def __init__(self, env, name: str, width: int, height: int) -> None:
        # Read `env.sim` HERE, never cache an MjSim across `env.reset()`:
        # robosuite's hard reset rebuilds the MjSim and `free()`s the old one, so
        # a cached handle fails with "'MjSim' object has no attribute 'model'".
        model = env.sim.model
        self.name = name
        self.width, self.height = width, height
        # Name lookup lives on robosuite's WRAPPER (`camera_name2id`); the raw
        # MjModel does not have it.
        # World-frame pose: `model.cam_pos` / `model.cam_mat0` are the camera's
        # LOCAL values inside its parent body. For `agentview` the parent is the
        # world, so local == world and the third-person numbers looked fine -- but
        # the wrist camera is mounted on the hand, where the locals are `(0.05, 0,
        # 0)` and every projection came out garbage (measured: all objects "out of
        # frame" with u=-89, v=-1129). `data.cam_xpos` / `data.cam_xmat` are the
        # refreshed world-frame values, which is also what robosuite's
        # get_camera_extrinsic_matrix uses.
        data = env.sim.data
        self.cam_id = int(model.camera_name2id(name))
        self.pos = np.asarray(data.cam_xpos[self.cam_id], dtype=float)
        self.mat = np.asarray(data.cam_xmat[self.cam_id], dtype=float).reshape(3, 3)
        self.fovy = math.radians(float(model.cam_fovy[self.cam_id]))
        self.fovx = 2.0 * math.atan((width / height) * math.tan(self.fovy / 2.0))

    def project(self, p_world: np.ndarray) -> tuple[float, float, float]:
        """world -> (u, v, perpendicular depth). Verified in probe_camera_projection."""
        p_cam = self.mat.T @ (np.asarray(p_world, dtype=float) - self.pos)
        depth = -p_cam[2]
        u_ndc = (p_cam[0] / depth) / math.tan(self.fovx / 2.0)
        v_ndc = (p_cam[1] / depth) / math.tan(self.fovy / 2.0)
        return (u_ndc + 1.0) / 2.0 * self.width, (1.0 - v_ndc) / 2.0 * self.height, depth

    def backproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        u_ndc = 2.0 * u / self.width - 1.0
        v_ndc = 1.0 - 2.0 * v / self.height
        p_cam = depth_m * np.array(
            [u_ndc * math.tan(self.fovx / 2.0), v_ndc * math.tan(self.fovy / 2.0), -1.0]
        )
        return self.pos + self.mat @ p_cam

    def mm_per_px(self, depth_m: float) -> float:
        return 2.0 * depth_m * math.tan(self.fovy / 2.0) / self.height * 1000.0


def depth_metres(env, cam: Camera) -> tuple[np.ndarray, float]:
    """Rendered depth in metres (OpenGL row order) + the far clip in metres."""
    model = env.sim.model
    extent = float(model.stat.extent)
    near = float(model.vis.map.znear) * extent
    far = float(model.vis.map.zfar) * extent
    # MEASURED shape of `sim.render(depth=True)`: a TUPLE (rgb uint8 (H,W,3),
    # depth float32 (H,W)) -- not the stereo pair first assumed, and not an array
    # (feeding the tuple to numpy raised "inhomogeneous shape").
    # MEASURED value range 0.9857..0.9962: MuJoCo's NORMALIZED depth -- the same
    # buffer robosuite's get_real_depth_map converts with near/(1-d*(1-near/far)),
    # which with near=0.011 m / far=530 m is effectively near/(1-d).
    rendered = env.sim.render(width=cam.width, height=cam.height, camera_name=cam.name, depth=True)
    norm = np.asarray(rendered[1] if isinstance(rendered, tuple) else rendered, dtype=np.float64)
    if norm.ndim == 3 and norm.shape == (2, cam.height, cam.width):
        norm = norm[0]
    norm = norm.reshape(cam.height, cam.width)
    metres = near / (1.0 - np.clip(norm, 0.0, 1.0) * (1.0 - near / far))
    return metres, far


def sample(depth_gl: np.ndarray, u: float, v: float, flip: bool) -> float:
    row = (depth_gl.shape[0] - 1 - int(round(v))) if flip else int(round(v))
    row = min(max(row, 0), depth_gl.shape[0] - 1)
    col = min(max(int(round(u)), 0), depth_gl.shape[1] - 1)
    return float(depth_gl[row, col])


def score_flip(
    cam: Camera, depth_gl: np.ndarray, points: dict[str, np.ndarray], flip: bool
) -> tuple[float, dict[str, float]]:
    """Median 3D distance between each GT point and the back-projection of its pixel."""
    errs: dict[str, float] = {}
    for name, gt in points.items():
        u, v, _ = cam.project(gt)
        if not (0 <= u < W and 0 <= v < H):
            continue
        errs[name] = float(np.linalg.norm(cam.backproject(u, v, sample(depth_gl, u, v, flip)) - gt))
    return (float(np.median(list(errs.values()))) if errs else float("inf")), errs


def plane_test(cam: Camera, depth_gl: np.ndarray, flip: bool, far: float) -> tuple[float, float, int]:
    """Back-project a 16x16 pixel grid; a FLAT table must come back flat.

    Returns (median z, spread = 1.4826*MAD, number of background samples). Under
    the wrong row order the grid samples the mirrored region -- largely
    background beyond the table -- which breaks coplanarity and shows up as
    samples sitting at the far clip.
    """
    zs: list[float] = []
    background = 0
    for u in np.linspace(8, W - 9, 16):
        for v in np.linspace(8, H - 9, 16):
            d = sample(depth_gl, u, v, flip)
            if d > far * 0.9:
                background += 1
                continue
            zs.append(float(cam.backproject(u, v, d)[2]))
    arr = np.asarray(zs)
    if arr.size == 0:
        return float("nan"), float("nan"), background
    spread = float(1.4826 * np.median(np.abs(arr - np.median(arr))))
    return float(np.median(arr)), spread, background


def report_camera(env, name: str, points: dict[str, np.ndarray], label: str) -> None:
    cam = Camera(env, name, W, H)
    depth_gl, far = depth_metres(env, cam)
    print(f"\n========= {label}：{name} =========")
    print(f"  cam pos={cam.pos.round(4)}  fovy={math.degrees(cam.fovy):.1f}°  "
          f"f={cam.height / (2 * math.tan(cam.fovy / 2)):.1f}px  far={far:.1f}m")

    print(f"  {'行序':>20s} {'表平面中位 z':>12s} {'平面散布(MAD)':>14s} {'背景采样':>9s}")
    for flip, tag in ((False, "row0=top"), (True, "row0=bottom(OpenGL)")):
        med, spread, bg = plane_test(cam, depth_gl, flip, far)
        print(f"  {tag:>20s} {med:12.4f} {spread * 1000:12.1f}mm {bg:8d}/256")

    print(f"  {'行序':>20s} {'中位 3D 距离':>12s}   逐点(mm)")
    scores = {}
    for flip, tag in ((False, "row0=top"), (True, "row0=bottom(OpenGL)")):
        med, errs = score_flip(cam, depth_gl, points, flip)
        scores[flip] = med
        detail = " ".join(f"{k.split('_')[0]}={v * 1000:.0f}" for k, v in errs.items())
        print(f"  {tag:>20s} {med * 1000:10.1f}mm   {detail}")
    flip = scores[True] < scores[False]
    print(f"  ⇒ 采用 {'row0=bottom (OpenGL)' if flip else 'row0=top'}")

    print(f"\n  {'point':20s} {'GT z':>7s} {'采样深度':>8s} {'中心深度':>8s} {'表面前后差':>10s} "
          f"{'mm/px':>6s} {'Δxy':>7s}")
    for pname, gt in sorted(points.items()):
        u, v, d_centre = cam.project(gt)
        if not (0 <= u < W and 0 <= v < H):
            print(f"  {pname:20s} 画面外 (u={u:.0f}, v={v:.0f})")
            continue
        d_surf = sample(depth_gl, u, v, flip)
        delta = cam.backproject(u, v, d_surf) - gt
        print(f"  {pname:20s} {gt[2]:7.4f} {d_surf:8.4f} {d_centre:8.4f} "
              f"{(d_centre - d_surf) * 1000:+9.1f}mm {cam.mm_per_px(d_surf):6.2f} "
              f"{np.linalg.norm(delta[:2]) * 1000:6.1f}mm")

    target = next((n for n in points if n.startswith("alphabet_soup")), None) or next(iter(points))
    gt = points[target]
    u0, v0, _ = cam.project(gt)
    inside = (0 <= u0 < W) and (0 <= v0 < H)
    d0 = sample(depth_gl, u0, v0, flip)
    back = cam.backproject(u0, v0, d0)
    u1, v1, _ = cam.project(back)
    print(f"\n  往返检验（{target}）：u {u0:.2f}->{u1:.2f}, v {v0:.2f}->{v1:.2f} "
          f"(差 {abs(u1 - u0):.3f}px / {abs(v1 - v0):.3f}px)")
    if inside:
        print(f"  决策粒度（depth={d0:.4f}m, {cam.mm_per_px(d0):.2f}mm/px）：")
        for du, dv in ((1, 0), (2, 0), (0, 1), (0, 2)):
            p = cam.backproject(u0 + du, v0 + dv, sample(depth_gl, u0 + du, v0 + dv, flip))
            print(f"    ±({du},{dv}) px -> 世界偏差 {np.linalg.norm((p - back)[:2]) * 1000:5.1f}mm")
        for pitch in GRID_PITCHES:
            su = (int(u0) // pitch) * pitch + pitch / 2
            sv = (int(v0) // pitch) * pitch + pitch / 2
            p = cam.backproject(su, sv, sample(depth_gl, su, sv, flip))
            print(f"    网格 {pitch:2d}px 吸附 -> {np.linalg.norm((p - back)[:2]) * 1000:6.1f}mm"
                  f"（像素搬了 {abs(su - u0):.0f},{abs(sv - v0):.0f}）")


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=W)
    adapter = ActionAdapter(sim)
    env = sim.env.env
    obs = sim.reset(init_state_idx=0)
    objects = {k: np.asarray(v, dtype=float) for k, v in obs["objects"].items()}
    eef = np.asarray(sim._obs[EEF_KEY], dtype=float)
    # The EEF site sits BETWEEN the pads, so a ray to it passes through the finger
    # gap and samples the table behind (measured at agentview: sampled depth
    # 1.517 m vs the site's own 1.102 m, i.e. a 414 mm "error" that is pure
    # geometry, not a projection bug). It is reported separately, never mixed into
    # the fidelity score.
    points = dict(objects)

    # The wrist camera at the HOME pose looks at the robot's own fingers (its
    # first "measurement" here sampled 4 cm of finger and produced a 145 mm
    # "error" plus the wrong row order). So it is measured TWICE: at home, and
    # after a descent to grasp height where it actually sees the can.
    report_camera(env, "agentview", points, "第三人称")
    report_camera(env, "robot0_eye_in_hand", points, "腕部（home 姿态）")

    can = objects["alphabet_soup_1"]
    u_eef, v_eef, d_eef = Camera(env, "agentview", W, H).project(eef)
    print(f"\n[ref] EEF site={eef.round(4)} → agentview 像素 ({u_eef:.1f},{v_eef:.1f})，"
          f"自身深度 {d_eef:.4f}m（两指间的点，不作保真度依据）")
    adapter.release()
    adapter.descend_to((can[0], can[1]), can[2] + 0.12)
    eef_low = np.asarray(sim._obs[EEF_KEY], dtype=float)
    print(f"\n[setup] 下移到抓取高度：eef={eef_low.round(4)}（罐心 z+0.12）")
    report_camera(env, "robot0_eye_in_hand", {**objects, "eef": eef_low}, "腕部（下降到抓取高度后）")

    sim.close()
    return 0


raise SystemExit(main())
