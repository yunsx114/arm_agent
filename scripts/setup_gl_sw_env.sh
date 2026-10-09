#!/usr/bin/env bash
# Create the `gl_sw` env: software EGL + Mesa drivers for headless rendering.
#
# WHY THIS EXISTS SEPARATELY FROM THE SIM ENV
#   The login node has no X display and no working hardware EGL:
#     * eglQueryDevicesEXT returns 2 devices;
#     * device 0 (the kms device) fails eglInitialize with EGL_NOT_INITIALIZED,
#       because /dev/dri/card0 is permission-denied;
#     * only device 1 works -> hence MUJOCO_EGL_DEVICE_ID=1 in simenv.sh.
#   So rendering goes through Mesa's software rasteriser, which is a *native
#   library* dependency, not a Python one. Putting it in its own env keeps it
#   out of the pinned libero env (robosuite needs numpy 1.22.x and mesa has no
#   opinion about numpy at all).
#
# Consumed via LD_LIBRARY_PATH / LIBGL_DRIVERS_PATH in scripts/simenv.sh.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

echo "==> creating conda env '$ENV_GL' (mesalib) in $CONDA_ROOT"
"$CONDA" create -y -n "$ENV_GL" -c conda-forge mesalib

echo
echo "==> verifying the libraries the renderer will dlopen"
for lib in libEGL.so.1 libGL.so.1; do
    if [ -e "$GL_LIB/$lib" ]; then
        echo "    OK   $GL_LIB/$lib"
    else
        echo "    MISS $GL_LIB/$lib" >&2
        exit 1
    fi
done
if [ -d "$GL_DRI" ]; then
    echo "    OK   $GL_DRI ($(ls "$GL_DRI" | tr '\n' ' '))"
else
    echo "    MISS $GL_DRI" >&2
    exit 1
fi

cat <<EOF

==> done. Nothing to install in Python; scripts/simenv.sh exports:
      LD_LIBRARY_PATH=$GL_LIB
      LIBGL_DRIVERS_PATH=$GL_DRI
      MUJOCO_GL=egl
      MUJOCO_EGL_DEVICE_ID=1
EOF
