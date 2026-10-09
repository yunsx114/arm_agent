# shellcheck shell=bash
# Shared path/env resolution for every script in this repo.
#
# WHY: every script here used to hard-code `/lab/haoq_lab/cse12311731`, which made
# the repo impossible to set up anywhere else and impossible to review (which
# path is the checkout and which is its parent?). The layout is actually
# positional, so it can be derived:
#
#   $ROOT/                     <- conda lives here, NOT in the repo
#   ├── miniconda3/envs/{libero,qwen35,gl_sw}
#   └── arm_agent/             <- this repo
#       ├── src/ scripts/ ...
#       └── third_party/LIBERO  <- cloned, .gitignore'd, pip install -e'd
#
# Override anything with env vars when your layout differs:
#   ARM_AGENT_ROOT=/data/foo   bash scripts/simenv.sh ...
#   CONDA_ROOT=/opt/conda      bash scripts/setup_agent_env.sh
#   ARM_AGENT_MODEL_DIR=/m/x   bash scripts/run_m1_on_gpu.sh 1

# shellcheck disable=SC2034  # consumed by the scripts that source this file
ARM_AGENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ARM_AGENT_ROOT:-$(dirname "$ARM_AGENT_DIR")}

CONDA_ROOT=${CONDA_ROOT:-$ROOT/miniconda3}
CONDA=$CONDA_ROOT/bin/conda

ENV_SIM=${ENV_SIM:-libero}
ENV_AGENT=${ENV_AGENT:-qwen35}
ENV_GL=${ENV_GL:-gl_sw}

PY_SIM=$CONDA_ROOT/envs/$ENV_SIM/bin/python
PY_AGENT=$CONDA_ROOT/envs/$ENV_AGENT/bin/python

# Give the two library paths as FINAL values, never as a root other scripts join
# path components onto. The first version of this file exposed
# `GL_LIB=$CONDA_ROOT/envs/$ENV_GL/lib` and the callers appended `/lib` again,
# producing `.../gl_sw/lib/lib/dri`. That path does not exist, Mesa then cannot
# open swrast_dri.so, and the failure surfaces as
#   ImportError: Cannot initialize a EGL device display
# from deep inside `import mujoco` -- i.e. a one-letter path bug that looks like
# a broken GPU stack. Keep them terminal.
GL_LIB=$CONDA_ROOT/envs/$ENV_GL/lib           # -> LD_LIBRARY_PATH
GL_DRI=$CONDA_ROOT/envs/$ENV_GL/lib/dri       # -> LIBGL_DRIVERS_PATH

# The model weights are 19GB and therefore live OUTSIDE the repo.
ARM_AGENT_MODEL_DIR=${ARM_AGENT_MODEL_DIR:-$ROOT/qwen35_demo/models/Qwen3.5-9B}

# Mirror + concurrency. `uv` is not optional in practice: pip downloads a single
# file per connection and the mirrors throttle one connection to ~150-250 KB/s,
# so the ~5GB agent env took >2h with pip and 22min with uv (measured).
PIP_INDEX=${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
UV=${UV:-$HOME/.local/bin/uv}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-16}

# ~/.local/lib/python3.9/site-packages shadows the libero env's pinned numpy
# (it holds 1.26.4, the env needs 1.22.4 for robosuite 1.4.0). Every entry point
# must set this; forgetting it produces a numpy ABI error deep inside robosuite.
export PYTHONNOUSERSITE=1

export_pythonpath() { PYTHONPATH=$ARM_AGENT_DIR/src${PYTHONPATH:+:$PYTHONPATH}; export PYTHONPATH; }
