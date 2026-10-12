"""Pixel -> world grounding, drawn from the camera and the depth buffer only.

WHY THIS MODULE EXISTS (档 B)
 The harness used to answer "where is the object" with the simulator's own body
 positions (`pack()["objects"]`), which `approach(<name>)` then consumed. That is
 ground truth, not perception: it cannot exist on a real robot, and it silently
 made every episode easier than the task. This module replaces that path with the
 one FBddcz/embodied-jev uses (`libero_policy.resolve_plan`, target kind
 "pixel"): the MODEL names a pixel in an image, and the code turns it into a world
 point via the camera's calibration and the rendered depth.

MEASURED CONSTRAINTS (scripts/probe_pixel_grounding.py, 2026-10-12)
  * depth buffer row order: row 0 is the BOTTOM of the scene (OpenGL). Decided by
    a 16x16 grid over a flat table: correct order returns the plane with a
    0.1-4.7 mm MAD, the wrong one with 34-526 mm.
  * the depth buffer is MuJoCo's NORMALIZED buffer: metres = near/(1-d*(1-near/far)).
  * a pixel is worth 2.96-4.04 mm from the third-person camera and 1.10 mm from
    the wrist camera 12 cm above the can -- so pointing at a pixel is a ~1 cm
    decision from the third-person view and a ~1 mm decision up close. Grasping
    needs ~2 mm, so the model is expected to use this tool twice (coarse, then
    close) rather than once.
  * the back-projected point lands on the VISIBLE SURFACE, +20..+90 mm in front of
    an object's centre (the basket's is +90 mm). That offset is exactly what the
    caller's `offset_mm` is for: clearance is the model's decision, not a constant
    baked into the tool.

Nothing here reads an object's name, position or size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Depth samples are taken as the median of a small neighbourhood: a single pixel
# can land on a mesh edge or an antialiased seam, where the depth jumps.
DEPTH_PATCH_RADIUS = 1
# Plausible depth range for this workspace (m). Outside it the sample is an
# error, not a target: the far clip reads ~530 m and a self-occluded pixel can
# read millimetres.
DEPTH_MIN_M = 0.02
DEPTH_MAX_M = 3.0


@dataclass
class Camera:
    """World-frame pinhole camera, refreshed from the live simulation state.

    `model.cam_pos` / `model.cam_mat0` are the camera's LOCAL values inside its
    parent body: correct for `agentview` (parent = world) but WRONG for the wrist
    camera (measured: locals (0.05, 0, 0), and every projection came out as
    "out of frame"). `data.cam_xpos` / `data.cam_xmat` are the world values, and
    are what robosuite's own get_camera_extrinsic_matrix reads.
    """

    name: str
    cam_id: int
    pos: np.ndarray
    mat: np.ndarray
    width: int
    height: int
    fovy: float
    fovx: float

    @classmethod
    def from_env(cls, env, name: str, width: int, height: int) -> "Camera":
        model = env.sim.model
        data = env.sim.data
        cam_id = int(model.camera_name2id(name))
        fovy = math.radians(float(model.cam_fovy[cam_id]))
        return cls(
            name=name,
            cam_id=cam_id,
            pos=np.asarray(data.cam_xpos[cam_id], dtype=float),
            mat=np.asarray(data.cam_xmat[cam_id], dtype=float).reshape(3, 3),
            width=width,
            height=height,
            fovy=fovy,
            fovx=2.0 * math.atan((width / height) * math.tan(fovy / 2.0)),
        )

    # ---------------------------------------------------------------- geometry
    def project(self, p_world: np.ndarray) -> tuple[float, float, float]:
        """world -> (u, v, perpendicular depth); u right, v down, origin top-left."""
        p_cam = self.mat.T @ (np.asarray(p_world, dtype=float) - self.pos)
        depth = -p_cam[2]
        u_ndc = (p_cam[0] / depth) / math.tan(self.fovx / 2.0)
        v_ndc = (p_cam[1] / depth) / math.tan(self.fovy / 2.0)
        return (u_ndc + 1.0) / 2.0 * self.width, (1.0 - v_ndc) / 2.0 * self.height, depth

    def backproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """(u, v, depth) -> world point on the surface the ray hits."""
        u_ndc = 2.0 * u / self.width - 1.0
        v_ndc = 1.0 - 2.0 * v / self.height
        p_cam = depth_m * np.array(
            [u_ndc * math.tan(self.fovx / 2.0), v_ndc * math.tan(self.fovy / 2.0), -1.0]
        )
        return self.pos + self.mat @ p_cam

    def mm_per_px(self, depth_m: float) -> float:
        """What one pixel is worth at this depth (the click-resolution budget)."""
        return 2.0 * depth_m * math.tan(self.fovy / 2.0) / self.height * 1000.0

    def inside(self, u: float, v: float) -> bool:
        return 0 <= u < self.width and 0 <= v < self.height


def depth_metres(env, camera: Camera) -> np.ndarray:
    """Rendered depth in metres, in OpenGL row order (row 0 = bottom).

    MEASURED: `sim.render(depth=True)` returns a TUPLE (rgb uint8 (H,W,3), depth
    float32 (H,W)) whose values are MuJoCo's normalized depth; converting with
    near/(1-d*(1-near/far)) reproduces a flat table to 0.1-4.7 mm.
    """
    model = env.sim.model
    extent = float(model.stat.extent)
    near = float(model.vis.map.znear) * extent
    far = float(model.vis.map.zfar) * extent
    rendered = env.sim.render(
        width=camera.width, height=camera.height, camera_name=camera.name, depth=True
    )
    norm = np.asarray(rendered[1] if isinstance(rendered, tuple) else rendered, dtype=np.float64)
    if norm.ndim == 3 and norm.shape == (2, camera.height, camera.width):
        norm = norm[0]
    norm = norm.reshape(camera.height, camera.width)
    return near / (1.0 - np.clip(norm, 0.0, 1.0) * (1.0 - near / far))


def sample_depth(
    depth_gl: np.ndarray, u: float, v: float, radius: int = DEPTH_PATCH_RADIUS
) -> float | None:
    """Median depth (metres) around a DISPLAY pixel, or None if unusable.

    Display pixel -> OpenGL row is `H - 1 - v`; getting that backwards would read
    the depth of a mirrored point, which is why scripts/probe_pixel_grounding.py
    pins it with a flat-table test.
    """
    height, width = depth_gl.shape
    col = int(round(u))
    row = (height - 1) - int(round(v))
    if not (0 <= col < width and 0 <= row < height):
        return None
    r0, r1 = max(row - radius, 0), min(row + radius + 1, height)
    c0, c1 = max(col - radius, 0), min(col + radius + 1, width)
    patch = depth_gl[r0:r1, c0:c1].reshape(-1)
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return None
    depth = float(np.median(patch))
    if not DEPTH_MIN_M < depth < DEPTH_MAX_M:
        return None
    return depth


@dataclass
class Grounding:
    """The full record of one pixel -> world decision (model-facing + numbers)."""

    camera: str
    pixel: tuple[int, int]
    depth_m: float
    surface_xyz: list[float]
    offset_m: list[float]
    target_xyz: list[float]
    mm_per_px: float
    grid_px: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def measurement(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "pixel": list(self.pixel),
            "depth_m": round(self.depth_m, 4),
            "mm_per_px": round(self.mm_per_px, 2),
            "surface_xyz": [round(v, 4) for v in self.surface_xyz],
            "offset_m": [round(v, 4) for v in self.offset_m],
            "target_xyz": [round(v, 4) for v in self.target_xyz],
        }


def ground_pixel(
    env, camera: Camera, u: float, v: float, offset_mm: list[float] | tuple[float, ...]
) -> tuple[Grounding | None, str]:
    """(pixel, depth) -> world target. Returns (grounding, reason) -- reason is
    non-empty exactly when grounding is None, so a caller can reject with a
    reason instead of silently substituting a different target."""
    if not camera.inside(u, v):
        return None, f"像素 ({u:.0f},{v:.0f}) 在 {camera.name} 画面之外"
    depth = sample_depth(depth_metres(env, camera), u, v)
    if depth is None:
        return None, (
            f"像素 ({u:.0f},{v:.0f}) 的深度不可用（超出 {DEPTH_MIN_M}~{DEPTH_MAX_M} m 或无效）"
        )
    surface = camera.backproject(u, v, depth)
    offset = np.asarray([float(x) for x in offset_mm], dtype=float) / 1000.0
    if offset.shape != (3,):
        return None, "offset_mm 需要三个数（世界系 x,y,z，单位毫米）"
    target = surface + offset
    return (
        Grounding(
            camera=camera.name,
            pixel=(int(round(u)), int(round(v))),
            depth_m=depth,
            surface_xyz=surface.tolist(),
            offset_m=offset.tolist(),
            target_xyz=target.tolist(),
            mm_per_px=camera.mm_per_px(depth),
        ),
        "",
    )


# ------------------------------------------------------------------ annotation
# Marker geometry for `aim` (档 B v2).
#
# The first proposal was a 3x3 dot: too small to judge a gap against when the model
# has to SEE its own error (measured 30-50 px, while one pixel is 3.7 mm at 256 px).
# The first implementation was a 6 px ring, and the first real aim run showed the other
# failure mode: the can is only ~16 px wide (8 px radius), so a 6 px ring overlapping
# its edge still reads as "the ring is on the can" -- the model stalled at 33-47 mm of
# error while reporting the ring as centred. 4 px keeps the pointer meaningfully
# smaller than the object it points at; the dark halo keeps it visible.
AIM_MARK_RADIUS_PX = 4
AIM_MARK_COLOUR = (255, 48, 48)
# The magnifier (档 B v2, added after the first aim run). MEASURED why it is needed:
# the model's ring corrections came in whole fine-cells (8-16 px) and it stopped at
# 9-13 px of error (33-47 mm), i.e. at the resolution the full frame affords. At 4x a
# 9 px error is 36 px of visible gap, so "is the ring's CENTRE on the object" stops
# being a judgement call. The crop is centred on whatever the caller is judging (the
# ring for `aim`, the gripper crosshair for `target`).
ZOOM_CROP_PX = 64
ZOOM_SCALE = 4


def crop_zoom(
    image_display: np.ndarray,
    centre: tuple[float, float],
    crop_px: int = ZOOM_CROP_PX,
    scale: int = ZOOM_SCALE,
) -> np.ndarray | None:
    """Magnify a window around `centre`, nearest-neighbour (no invented pixels).

    The window is SHIFTED to stay inside the frame rather than padded: padding would
    put a black band where the world continues, and a model reading that band as
    "nothing there" is exactly the class of quiet lie this module avoids. Returns
    None when the frame is smaller than the window, so a caller can omit the image
    instead of shipping a broken one.
    """
    from PIL import Image

    arr = np.asarray(image_display)
    height, width = arr.shape[:2]
    size = int(min(crop_px, width, height))
    if size < 8:
        return None
    u, v = float(centre[0]), float(centre[1])
    left = round(min(max(u - size / 2.0, 0.0), width - size))
    top = round(min(max(v - size / 2.0, 0.0), height - size))
    crop = Image.fromarray(arr[top : top + size, left : left + size]).convert("RGB")
    return np.asarray(crop.resize((size * int(scale), size * int(scale)), Image.NEAREST))


def annotate_frame(
    image_display: np.ndarray,
    camera: Camera,
    eef_world: np.ndarray | None,
    grid_px: int,
    minor_px: int | None = None,
    mark: tuple[float, float] | None = None,
    mark_radius: int | None = None,
) -> np.ndarray:
    """Draw a coordinate grid, the gripper's own projected pixel, and (only when
    `aim` asks for it) a ring where the MODEL said it wanted to go.

    Only two things go on the image by default: a numbered grid (so the model can
    NAME a pixel) and a crosshair where the gripper actually is. No object labels,
    no boxes -- identifying the target is the job.

    `mark` is the 档 B v2 addition and it changes what the image is FOR. The
    crosshair answers "where is my hand"; the ring answers "where would I land if I
    asked now" -- a dry run with no motion. Comparing those two against the object
    is a RELATIVE judgement ("the ring is left of the can by two cells"), which is
    the thing a small VLM can do, unlike naming an absolute pixel (measured: 30-50 px
    systematic bias at both 256 px and 512 px, stable across four calls).

    The ring is drawn at the pixel the caller ASKED for, never at the pixel that
    the requested point re-projects to: the round trip is 0.000 px (measured in
    probe_pixel_grounding), so re-projecting would hide exactly the grounding error
    this tool exists to expose.

    MEASURED grid cost (scripts/probe_pixel_grounding.py, third-person view at the
    can's depth, 3.72 mm/px): snapping to a 4 px cell costs 2.8 mm, 8 px costs
    9.5 mm, 16 px costs 32 mm, 32 px costs 185 mm. So the LABELS only have to
    appear every `grid_px` (32 keeps the numbers readable), while the FINE
    structure the model counts against must be much closer than that. First
    version drew only 32 px lines and the model's first real run aimed 30-50 px
    off the target (11-19 cm) -- the grid it had to count on was 185 mm coarse.
    Minor lines are therefore drawn at `grid_px // 4` (8 px by default).
    """
    from PIL import Image, ImageDraw

    pil = Image.fromarray(np.asarray(image_display)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    width, height = pil.size
    grid_px = max(int(grid_px), 8)
    minor = max(int(minor_px if minor_px else grid_px // 4), 2)
    for x in range(minor, width, minor):
        if x % grid_px:
            draw.line((x, 0, x, height), fill=(150, 158, 164), width=1)
    for y in range(minor, height, minor):
        if y % grid_px:
            draw.line((0, y, width, y), fill=(150, 158, 164), width=1)
    for x in range(0, width, grid_px):
        draw.line((x, 0, x, height), fill=(96, 106, 116), width=1)
        draw.text((x + 1, 1), str(x), fill=(255, 255, 255))
    for y in range(grid_px, height, grid_px):
        draw.line((0, y, width, y), fill=(96, 106, 116), width=1)
        draw.text((1, y + 1), str(y), fill=(255, 255, 255))
    if eef_world is not None:
        u, v, _ = camera.project(np.asarray(eef_world, dtype=float))
        if camera.inside(u, v):
            draw.line((u - 8, v, u + 8, v), fill=(0, 255, 255), width=2)
            draw.line((u, v - 8, u, v + 8), fill=(0, 255, 255), width=2)
    if mark is not None:
        mu, mv = float(mark[0]), float(mark[1])
        # An off-frame mark is NOT clamped to the border: the caller rejects the
        # request with a reason instead, and a dot drawn at the edge would read as
        # "your point is here", i.e. as a silently different target (Jev's rule).
        if camera.inside(mu, mv):
            r = max(int(mark_radius or AIM_MARK_RADIUS_PX), 2)
            # Halo first, ring on top: the halo keeps it visible on a white object,
            # the 1 px ring keeps the object's edge readable where it crosses.
            draw.ellipse((mu - r, mv - r, mu + r, mv + r), outline=(0, 0, 0), width=3)
            draw.ellipse((mu - r, mv - r, mu + r, mv + r), outline=AIM_MARK_COLOUR, width=1)
            draw.rectangle((mu - 2, mv - 2, mu + 2, mv + 2), fill=(0, 0, 0))
            draw.rectangle((mu - 1, mv - 1, mu + 1, mv + 1), fill=AIM_MARK_COLOUR)
    return np.asarray(pil)
