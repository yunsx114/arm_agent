"""Phase 0 · E3 probe: can a GPU node (gpu026) run MuJoCo physics at all?

Context: gpu026 is a VM whose CPUID flags are cropped (reported: no AVX/AVX2),
and `import mujoco` previously SIGILL'd there for the a3 project. But this node
is the ONLY place the 4bit LLM can run, so if it can also run MuJoCo physics we
can collapse to a single-node design (Plan A).

Run via srun (see job.e3_mujoco.sh). Prints cpuinfo flags, import/step timing,
and whether EGL rendering is available with the NVIDIA driver.

Exit codes: 0 = physics works, 2 = SIGILL/illegal instruction on import,
3 = physics OK but rendering failed.
"""

from __future__ import annotations

import platform
import time


def cpu_flags() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "<unreadable>"


def main() -> int:
    flags = cpu_flags()
    print("[e3] host:", platform.node())
    print("[e3] cpu flags:", flags[:200], "...")
    for want in ("avx", "avx2", "avx512f", "sse4_2", "ssse3"):
        print(f"[e3] has {want}:", want in flags.split())

    import os

    print("[e3] MUJOCO_GL =", os.environ.get("MUJOCO_GL"))

    try:
        import mujoco
    except Exception as exc:  # noqa: BLE001 - probe
        print(f"[e3] FAIL import mujoco: {type(exc).__name__}: {exc}")
        return 2

    print("[e3] mujoco", mujoco.__version__)

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/>"
        "<geom type='box' size='.05 .05 .05'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    t0 = time.time()
    for _ in range(500):
        mujoco.mj_step(model, data)
    dt = time.time() - t0
    print(f"[e3] 500 physics steps in {dt * 1000:.0f} ms ({dt / 500 * 1e6:.0f} us/step)")
    print("[e3] PHYSICS OK")

    # Rendering is a bonus on this node (NVIDIA EGL). a3 measured SIGILL on
    # import; if physics imported fine, try the renderer too.
    try:
        renderer = mujoco.Renderer(model, 128, 128)
        renderer.update_scene(data)
        img = renderer.render()
        print(f"[e3] render OK mean={img.mean():.1f}")
        return 0
    except Exception as exc:  # noqa: BLE001 - probe
        print(f"[e3] render FAILED: {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
