"""Phase 0 · independent check of the image-axis mapping used for the E6 ground truth.

E6's multiple-choice answers were derived from a HUMAN reading of the rendered
image ("y+ points away from the camera"). That is exactly the kind of
hand-derived assumption that has produced wrong tables before, so this probe
verifies it mechanically: it projects each object's world position through the
agentview camera's actual extrinsics/intrinsics and reports pixel coordinates.

If the projection says alphabet_soup is at the bottom of the image and the
basket at the top, the E6 GT holds; otherwise the E6 answers must be rewritten.

Reference maths (MuJoCo convention): cameras look down their local -Z, +Y is up
in image space. cam_mat0 is the world<-camera rotation, row-flattened (9 values,
row-major in the binding). So:

    p_cam = R^T (p_world - cam_pos)
    u_ndc = (p_cam.x / -p_cam.z) / tan(fovx/2)
    v_ndc = (p_cam.y / -p_cam.z) / tan(fovy/2)
    u_px  = (u_ndc + 1)/2 * W          v_px = (1 - v_ndc)/2 * H   (v=0 at TOP)

The saved PNG in outputs/smoke is the display-oriented image (LIBERO stores it
vertically flipped, saved as [::-1]), and this formula yields coordinates in
that same orientation: v_px 0 = top of the saved PNG.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCENE_JSON = REPO / "outputs" / "smoke" / "e1_scene_state.json"


def main() -> int:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    W = H = 256
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=H, camera_widths=W)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    model = env.env.sim.model
    # find the agentview camera id
    cam_id = None
    for i in range(model.ncam):
        if model.camera_id2name(i) == "agentview":
            cam_id = i
            break
    assert cam_id is not None, "agentview camera not found"

    cam_pos = np.array(model.cam_pos[cam_id])
    R = np.array(model.cam_mat0[cam_id]).reshape(3, 3)  # world<-camera
    fovy_deg = float(model.cam_fovy[cam_id])
    fovy = math.radians(fovy_deg)
    fovx = 2 * math.atan((W / H) * math.tan(fovy / 2))

    print(f"[proj] agentview cam id={cam_id} pos={cam_pos.round(3)} fovy={fovy_deg:.1f}deg")

    def project(p_world: np.ndarray) -> tuple[float, float, float]:
        p_cam = R.T @ (p_world - cam_pos)
        depth = -p_cam[2]
        u_ndc = (p_cam[0] / depth) / math.tan(fovx / 2)
        v_ndc = (p_cam[1] / depth) / math.tan(fovy / 2)
        return (u_ndc + 1) / 2 * W, (1 - v_ndc) / 2 * H, depth

    # cross-check the empirical axis mapping claimed in E6's docstring
    gt = json.loads(SCENE_JSON.read_text())
    eef = np.asarray(gt["eef_pos"])
    print("\n[proj] object projections (u=0 left, v=0 TOP of the saved PNG):")
    print(f"{'object':22s} {'world (x,y,z)':34s} {'u_px':>7s} {'v_px':>7s} {'depth':>7s}")
    rows = []
    for name, pos in sorted(gt["objects"].items()):
        if name.endswith("_to_robot0_eef"):
            continue
        p = np.asarray(pos)
        u, v, d = project(p)
        rows.append((name, p, u, v, d))
        print(f"{name:22s} {str(p.round(3)):34s} {u:7.1f} {v:7.1f} {d:7.3f}")

    u_eef, v_eef, d_eef = project(eef)
    print(f"{'ROBOT0_EEF(arm tip)':22s} {str(eef.round(3)):34s} {u_eef:7.1f} {v_eef:7.1f} {d_eef:7.3f}")

    # --- mechanically re-derive the E6 answers ---
    print("\n[proj] --- mechanically derived E6 answers ---")
    obj_xy = {n: p for n, p, *_ in rows}

    soup = obj_xy["alphabet_soup_1"]
    dy_soup = soup[1] - eef[1]
    print(f"Q1 direction of alphabet_soup rel. to eef: dx={soup[0]-eef[0]:+.3f} dy={dy_soup:+.3f}"
          f" -> {'FAR' if dy_soup > 0 else 'NEAR'} side, {'RIGHT' if soup[0]-eef[0] > 0 else 'LEFT'}")

    basket = obj_xy["basket_1"]
    print(f"Q2 basket rel. to eef: dx={basket[0]-eef[0]:+.3f} dy={basket[1]-eef[1]:+.3f}"
          f" -> {'FAR' if basket[1]-eef[1] > 0 else 'NEAR'} side")

    nearest = min((r for r in rows if r[0] != "ROBOT0_EEF(arm tip)"), key=lambda r: r[1][1])
    print(f"Q4 nearest-to-camera object (min y): {nearest[0]} (y={nearest[1][1]:.3f})")

    # vertical ordering in the image must agree with the y-ordering
    print("\n[proj] consistency check: is v_px ordering the same as y ordering?")
    by_v = sorted(rows, key=lambda r: r[3])
    by_y = sorted(rows, key=lambda r: r[1][1])
    print("  top->bottom by v_px:", [r[0] for r in by_v])
    print("  near...far by y   :", [r[0] for r in by_y])
    consistent = [r[0] for r in by_v] == [r[0] for r in by_y]
    print(f"  {'CONSISTENT: larger y = higher in image' if consistent else 'INCONSISTENT - E6 GT is WRONG'}")

    env.close()
    return 0 if consistent else 1


if __name__ == "__main__":
    sys.exit(main())
