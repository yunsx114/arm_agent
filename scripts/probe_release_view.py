"""Reproduce the exact pose the model was in when it released the can, and dump
the two 256 px frames it actually received (no recorder overlay).

Question this answers (asked 2026-10-10): at the release in
episode_000_1791623248 turn 31, was the basket's opening visible under the pads
in the wrist view -- i.e. did the harness give the model what rule A asks it to
check ("confirm from both frames that the container's mouth is directly below the
pads"), or is that check impossible from the supplied images?

Method: replay the measured transport from the transcript --
    t2  descend_to(alphabet soup, 1.2)   -> eef (-0.1527, -0.2397, ...)
    t3  gripper close                    -> width 0.0315 (grasped)
    t4  move +z 5
    t5  move +y 10, t6 +5, t7 +5         -> eef y = 0.2921, z = 0.0979
then save sim.images() byte-for-byte at cam_size 256 (what llm.py feeds).

Also prints the basket opening's world AABB, so "was the release point inside the
mouth" is arithmetic rather than eyeballing.

Run: cd arm_agent && bash scripts/simenv.sh scripts/probe_release_view.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.sim.action_adapter import ActionAdapter  # noqa: E402
from arm_agent.sim.libero_env import LiberoSim  # noqa: E402

OUT = Path("/root/arm_agent/outputs/smoke/release_view")


def geom_world_aabb(env, body_filter: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact world AABB of every geom of the named body.

    Uses MESH VERTICES when the geom is a mesh. The first version fell back to
    a bounding sphere (max geom_size) for meshes, which reported a 29 x 30 cm
    "basket" -- physically impossible for LIBERO's basket and useless for
    judging whether a release point is inside the mouth. mjGEOM_BOX/CAPSULE are
    handled analytically; there is no sphere fallback any more.
    """
    import mujoco

    sim = env.sim
    sim.forward()
    model, data = sim.model, sim.data
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    hit = False
    for gid in range(model.ngeom):
        bname = model.body_id2name(model.geom_bodyid[gid]) or ""
        if body_filter not in bname.lower():
            continue
        hit = True
        gt = model.geom_type[gid]
        gsz = np.asarray(model.geom_size[gid], dtype=float)
        if gt == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[gid])
            a0, a1 = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            pts = np.asarray(model.mesh_vert[a0 : a0 + a1], dtype=float)
        elif gt == int(mujoco.mjtGeom.mjGEOM_BOX):
            pts = np.array(
                [[sx * gsz[0], sy * gsz[1], sz * gsz[2]]
                 for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            )
        elif gt in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
            r = float(gsz[0])
            half = float(gsz[1]) if gt == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0
            pts = np.array(
                [[sx * r, sy * r, sz * (r + half)]
                 for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            )
        else:
            continue
        world = data.geom_xpos[gid] + pts @ data.geom_xmat[gid].reshape(3, 3).T
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    return (lo, hi) if hit else None


def main() -> int:
    sim = LiberoSim(suite_name="libero_object", task_id=0, cam_size=256)
    adapter = ActionAdapter(sim)

    obs = sim.reset(init_state_idx=0)
    objs = obs["objects"]
    ckey = [k for k in objs if "soup" in k][0]
    bkey = [k for k in objs if "basket" in k][0]
    can0 = np.asarray(objs[ckey], dtype=float)
    basket = np.asarray(objs[bkey], dtype=float)
    print(f"can    at ({can0[0]:+.4f}, {can0[1]:+.4f}, {can0[2]:+.4f})")
    print(f"basket at ({basket[0]:+.4f}, {basket[1]:+.4f}, {basket[2]:+.4f})")
    box = geom_world_aabb(sim.env, "basket")
    if box is not None:
        lo, hi = box
        print(f"basket geoms world AABB: x[{lo[0]:+.4f},{hi[0]:+.4f}] "
              f"y[{lo[1]:+.4f},{hi[1]:+.4f}] z[{lo[2]:+.4f},{hi[2]:+.4f}]")
        print(f"  opening (x,y) half-extents: {(hi[0]-lo[0])/2*100:.1f} x "
              f"{(hi[1]-lo[1])/2*100:.1f} cm; centre "
              f"({(lo[0]+hi[0])/2:+.4f}, {(lo[1]+hi[1])/2:+.4f})")

    # --- replay the measured transport ------------------------------------
    o = adapter.descend_to((can0[0], can0[1]), can0[2] + 0.012)
    print(f"\nreplay descend_to(can, 1.2) -> {o.note}")
    o = adapter.set_gripper("close")
    print(f"replay close                -> width {adapter._gripper_now():.4f}")
    adapter.move("+z", 5.0)
    # 0.24 m of +y at 5 cm/step needs ~11 steps: the first version looped only
    # four times and silently stopped at y = -0.04, so the dumped frames showed
    # the milk carton and no basket at all. Bounded by distance, not by guess.
    for _step in range(14):
        eef = adapter._eef_now()
        if eef[1] >= 0.2921:
            break
        adapter.move("+y", min(5.0, (0.2921 - eef[1]) * 100))
        eef2 = adapter._eef_now()
        print(f"replay move +y {min(5.0, (0.2921 - eef[1]) * 100):4.1f}"
              f"     -> y {eef[1]:+.4f} -> {eef2[1]:+.4f}")
    # settle to the measured release height
    eef = adapter._eef_now()
    if abs(eef[2] - 0.0979) > 0.002:
        adapter.move("+z" if eef[2] < 0.0979 else "-z", abs(eef[2] - 0.0979) * 100)
    eef = adapter._eef_now()
    print(f"\nreplayed release pose: ({eef[0]:+.4f}, {eef[1]:+.4f}, {eef[2]:+.4f})")
    print("measured t31 release : (-0.1388, 0.2921, 0.0979)")

    if box is not None:
        lo, hi = box
        inside_x = lo[0] <= eef[0] <= hi[0]
        inside_y = lo[1] <= eef[1] <= hi[1]
        print(f"  eef x inside basket x-range? {inside_x}   "
              f"off by {max(lo[0]-eef[0], 0, eef[0]-hi[0])*100:.1f} cm")
        print(f"  eef y inside basket y-range? {inside_y}   "
              f"off by {max(lo[1]-eef[1], 0, eef[1]-hi[1])*100:.1f} cm")
        print(f"  horizontal distance to basket centre: "
              f"{np.linalg.norm(eef[:2] - basket[:2])*100:.1f} cm")
        print(f"  horizontal distance to nearest opening edge: "
              f"{np.hypot(max(lo[0]-eef[0], 0, eef[0]-hi[0]), max(lo[1]-eef[1], 0, eef[1]-hi[1]))*100:.1f} cm")

    can_now = np.asarray(sim.pack(sim._obs)["objects"][ckey], dtype=float)
    print(f"can while held: ({can_now[0]:+.4f}, {can_now[1]:+.4f}, {can_now[2]:+.4f})  "
          f"(can-EEF x offset {can_now[0]-eef[0]:+.4f}, y offset {can_now[1]-eef[1]:+.4f})")

    # --- the frames the model actually receives ---------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sim.images()
    from PIL import Image

    for cam, arr in imgs.items():
        Image.fromarray(np.asarray(arr)).save(OUT / f"release_{cam}.png")
        print(f"saved {OUT / f'release_{cam}.png'}  shape {np.asarray(arr).shape}")

    # --- the decisive test: actually RELEASE here and let it fall -----------
    # Same action the model took at t31, at the same measured pose. The sim is
    # the only authority on "would this release have delivered the can?".
    print("\n=== drop test: open the gripper at this pose, then settle ===")
    adapter.set_gripper("open")
    zero = np.zeros(sim.env.action_dim, dtype=float)
    for i in range(30):
        sim.step(zero)
        if i in (0, 4, 9, 19, 29):
            can_now = np.asarray(sim.pack(sim._obs)["objects"][ckey], dtype=float)
            print(f"  after {i + 1:>2} env steps: can "
                  f"({can_now[0]:+.4f}, {can_now[1]:+.4f}, {can_now[2]:+.4f})  "
                  f"success={sim.check_success()}")
    can_end = np.asarray(sim.pack(sim._obs)["objects"][ckey], dtype=float)
    print(f"  FINAL can    ({can_end[0]:+.4f}, {can_end[1]:+.4f}, {can_end[2]:+.4f})")
    print(f"  basket centre({basket[0]:+.4f}, {basket[1]:+.4f}, {basket[2]:+.4f})")
    print(f"  horizontal miss from basket centre: "
          f"{np.linalg.norm(can_end[:2] - basket[:2]) * 100:.2f} cm")
    print(f"  check_success() = {sim.check_success()}")
    if box is not None:
        lo, hi = box
        print(f"  can inside opening rect? x {lo[0] - 0.033 <= can_end[0] <= hi[0] + 0.033} "
              f"y {lo[1] - 0.033 <= can_end[1] <= hi[1] + 0.033} "
              f"(can radius 3.3 cm added to the rect)")
    imgs2 = sim.images()
    for cam, arr in imgs2.items():
        Image.fromarray(np.asarray(arr)).save(OUT / f"after_drop_{cam}.png")
    print(f"saved {OUT}/after_drop_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
