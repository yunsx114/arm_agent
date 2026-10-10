#!/usr/bin/env bash
# Run a command inside the `libero` sim env with the exact runtime variables the
# login node needs. Usage:
#
#   bash scripts/simenv.sh scripts/smoke_libero_e1.py
#   bash scripts/simenv.sh -c "import libero; print('ok')"
#   bash scripts/simenv.sh            # interactive python in the sim env
#
# Variables and why:
#   PYTHONNOUSERSITE=1  -> ~/.local/lib/python3.9 holds numpy 1.26.4 which
#                          shadows the env's pinned 1.22.4 (breaks robosuite).
#   LD_LIBRARY_PATH     -> gl_sw mesalib provides libEGL.so.1 (software EGL).
#   LIBGL_DRIVERS_PATH  -> gl_sw/lib/dri holds kms_swrast_dri.so.
#   MUJOCO_GL=egl       -> force headless EGL (no X display on this node).
#   MUJOCO_EGL_DEVICE_ID -> measured on login01: eglQueryDevicesEXT returns 2
#                          devices; device 0's eglInitialize fails with
#                          EGL_NOT_INITIALIZED (it is the kms device and
#                          /dev/dri/card0 is permission-denied), device 1 works.
#                          robosuite's EGLGLContext tries ONLY the indexed device
#                          (mujoco.Renderer loops over all, which is why the pure
#                          mujoco path worked without this), so we must point it
#                          at device 1 explicitly.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

export PYTHONPATH=$ARM_AGENT_DIR/src:$ARM_AGENT_DIR/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}
# The original cluster needed Mesa software EGL on its login node.  A local
# NVIDIA node should use the driver's EGL implementation instead; prepending
# Mesa here makes MuJoCo select the wrong platform and fail on RTX 5090.
SIM_EGL_MODE=${SIM_EGL_MODE:-hardware}
if [ "$SIM_EGL_MODE" = "software" ]; then
    export LD_LIBRARY_PATH=$GL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export LIBGL_DRIVERS_PATH=$GL_DRI
    export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-1}
elif [ "$SIM_EGL_MODE" = "hardware" ]; then
    unset LIBGL_DRIVERS_PATH
    export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
else
    echo "SIM_EGL_MODE must be 'hardware' or 'software', got: $SIM_EGL_MODE" >&2
    exit 2
fi
export MUJOCO_GL=egl
# LIBERO stores trusted NumPy-backed init states as legacy torch pickles.
# PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects them.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

exec "$PY_SIM" "$@"
