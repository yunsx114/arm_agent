#!/usr/bin/env bash
# Create the `libero` conda env for the simulation side of arm_agent.
#
# Verified facts this script encodes (2026-10-07, login01):
#  - robosuite==1.4.0 is what LIBERO's env code targets; it bundles a 193MB
#    wheel that runs with mujoco>=2.3.0. We pin mujoco==2.3.7 (the 2.3.x line
#    that shipped alongside robosuite 1.4.0).
#  - Do NOT install torch/lerobot/transformers here: LIBERO's sim path is pure
#    numpy + MuJoCo. Datasets (several GB) are only needed for training.
#  - ~/.local/lib/python3.9/site-packages shadows the env's numpy (it holds
#    1.26.4 and wins over the env's 1.22.4). Always run with PYTHONNOUSERSITE=1.
#  - Rendering on the login node uses software EGL from the `gl_sw` env
#    (mesalib). See scripts/simenv.sh for the runtime env vars.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

UV=${UV:-$HOME/.local/bin/uv}
PY=$PY_SIM

"$CONDA" create -n "$ENV_SIM" python=3.9 -y

export UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=16

# Core sim stack. numpy MUST stay in the 1.2x line for robosuite 1.4.0.
"$UV" pip install --python "$PY" --index-url https://mirrors.aliyun.com/pypi/simple/ \
  "robosuite==1.4.0" \
  "mujoco==2.3.7" \
  "bddl==1.0.1" \
  "numpy==1.22.4" \
  "opencv-python-headless==4.8.1.78" \
  "gym==0.25.2" \
  "easydict" \
  "matplotlib" \
  "termcolor" \
  "cloudpickle" \
  "future" \
  "pyyaml" \
  "pillow"

echo
echo "Verifying imports (needs gl_sw EGL for the renderer import path)..."
env PYTHONNOUSERSITE=1 \
    LD_LIBRARY_PATH=$GL_LIB \
    LIBGL_DRIVERS_PATH=$GL_DRI \
    MUJOCO_GL=egl \
    "$PY" -c "
import numpy, mujoco, robosuite, bddl
assert numpy.__version__.startswith('1.2'), numpy.__version__
print('numpy', numpy.__version__)
print('mujoco', mujoco.__version__)
print('robosuite', robosuite.__version__)
print('OK: libero env ready')
"

echo
echo "Next: bash scripts/setup_env.sh --only libero   (clones + installs LIBERO)"
echo "Then run with:  bash scripts/simenv.sh scripts/smoke_libero_e1.py"
