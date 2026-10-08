"""LIBERO environment wrapper: task loading, reset, stepping, observation packing.

Lives in the `libero` conda env (python 3.9) and runs on the LOGIN node (the
gpu026 CPU cannot execute MuJoCo: no SSE4.2/AVX -> SIGILL, see DESIGN.md E3).

Everything the rest of the system needs to know about a step comes out of
`pack_observation`, which is also what crosses the NFS IPC boundary.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from arm_agent import contracts as C


class LiberoSim:
    """One LIBERO task instance with deterministic init states."""

    def __init__(
        self,
        suite_name: str = "libero_object",
        task_id: int = 0,
        cam_size: int = 256,
        seed: int = 0,
    ) -> None:
        # Imported lazily so that importing this module on machines without
        # EGL/LIBERO (e.g. an agent-side process) does not explode.
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.suite_name = suite_name
        self.task_id = task_id
        self.cam_size = cam_size

        self.suite = benchmark.get_benchmark_dict()[suite_name]()
        self.task = self.suite.get_task(task_id)
        bddl = os.path.join(
            get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file
        )
        # ignore_done=True (the official LIBERO eval mode): robosuite otherwise
        # flips self.done at timestep >= horizon(1000) and then REFUSES further
        # steps with "executing action in terminated episode" — measured the
        # hard way when a probe's greedy loop spent >1000 ticks against an
        # unreachable goal. Episode length is managed by the harness instead.
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=cam_size,
            camera_widths=cam_size,
            ignore_done=True,
        )
        self.env.seed(seed)
        self.init_states = self.suite.get_task_init_states(task_id)

        self._obs: dict[str, Any] | None = None
        self.step_count = 0
        self.success = False
        # Rendering is 90% of a 245 ms step (measured, probe_render_cost.py):
        # internal closed-loop ticks run with cameras disabled and refresh once
        # per tool call.
        self._render_enabled = True
        self._render_dirty = True

    # ------------------------------------------------------------- rendering
    def set_render(self, enabled: bool) -> None:
        """Toggle camera observables (skips their per-step sensor evaluation)."""
        if enabled == self._render_enabled:
            return
        for name, observable in self.env.env._observables.items():
            if "image" in name:
                observable.set_enabled(enabled)
        self._render_enabled = enabled
        if not enabled:
            self._render_dirty = True

    def refresh_images(self) -> None:
        """Re-render cameras at the current physics state (idempotent)."""
        if not self._render_dirty and self._render_enabled:
            return
        self.set_render(True)
        self._obs = self.env.env._get_observations(force_update=True)
        self._render_dirty = False

    # ------------------------------------------------------------------ reset
    def reset(self, init_state_idx: int = 0) -> dict[str, Any]:
        """Reset to a benchmark init state and settle physics (official flow).

        The 5 zero-action steps are the LIBERO demo convention: physics settles
        and the gripper stays open, so every episode starts from the same
        measured configuration.
        """
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[init_state_idx])
        self.set_render(False)
        for _ in range(C.OBS_SETTLE_STEPS):
            obs, _reward, _done, _info = self.env.step(np.zeros(C.ACTION_DIM))
        # Order matters: settle first (its obs has no frames because the
        # cameras were off), then refresh once; refresh_images() writes the
        # full obs with frames into self._obs.
        self._obs = obs
        self.refresh_images()
        self.step_count = 0
        self.success = self.check_success()
        return self.pack(self._obs)

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray | list[float]) -> dict[str, Any]:
        """One control tick (20 Hz). Returns the packed observation."""
        action = np.asarray(action, dtype=np.float64).reshape(C.ACTION_DIM)
        obs, _reward, _done, _info = self.env.step(action)
        self._obs = obs
        self.step_count += 1
        self.success = self.check_success()
        return self.pack(obs)

    def check_success(self) -> bool:
        return bool(self.env.check_success())

    # ------------------------------------------------------------------- pack
    def pack(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Pack the raw 40-key observation into the IPC-friendly shape.

        Images stay as numpy arrays here; the server turns them into PNG files.
        GT object positions are included deliberately: they back the optional
        `locate` tool and are used by evaluation/debugging, but the harness may
        choose not to expose them to the model.
        """
        objects = {}
        for key, value in obs.items():
            if (
                key.endswith("_pos")
                and not key.startswith(C.ROBOT_PREFIX)
                and not key.endswith("_to_robot0_eef_pos")
            ):
                objects[key[: -len("_pos")]] = np.asarray(value).round(5).tolist()

        return {
            "task": self.task.name,
            "language": self.task.language,
            "step_count": self.step_count,
            "success": self.success,
            "eef_pos": np.asarray(obs[C.OBS_EEF_POS]).round(5).tolist(),
            "eef_quat": np.asarray(obs[C.OBS_EEF_QUAT]).round(5).tolist(),
            "gripper_qpos": np.asarray(obs[C.OBS_GRIPPER_QPOS]).round(5).tolist(),
            "objects": objects,
        }

    # ---------------------------------------------------------------- images
    def images(self) -> dict[str, np.ndarray]:
        """Latest camera frames (display orientation; see probe_camera_projection)."""
        assert self._obs is not None, "call reset() first"
        return {
            "agent": np.asarray(self._obs[C.OBS_AGENTVIEW]),
            "wrist": np.asarray(self._obs[C.OBS_WRIST]),
        }

    def close(self) -> None:
        self.env.close()
