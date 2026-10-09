#!/usr/bin/env bash
# One-shot environment setup for arm_agent. Idempotent: re-running skips what
# already exists, so it doubles as a health check.
#
#   bash scripts/setup_env.sh                 # everything
#   bash scripts/setup_env.sh --only libero   # one stage
#   bash scripts/setup_env.sh --check         # verify only, install nothing
#   FORCE=1 bash scripts/setup_env.sh --only agent   # recreate that env
#
# Stages, in order (each is a separate script you can also run alone):
#   1. gl_sw   scripts/setup_gl_sw_env.sh    software EGL for headless rendering
#   2. libero  scripts/setup_sim_env.sh      MuJoCo + robosuite + LIBERO
#   3. agent   scripts/setup_agent_env.sh    torch + transformers + 4bit
#   4. model   scripts/download_model.sh     19GB weights (skipped if present)
#
# WHY SO MANY ENVS -- this is not gold-plating, the pins genuinely conflict:
#   * libero needs numpy 1.22.4 (robosuite 1.4.0 / mujoco 2.3.7), python 3.9;
#   * qwen35 needs python 3.11 + torch 2.8 (sm_75) and cannot run MuJoCo at all;
#   * gl_sw is native library only, no python packages worth pinning.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

ONLY=""
CHECK_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY=${2:?--only needs a stage name}; shift 2 ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
env_exists() { [ -x "$CONDA_ROOT/envs/$1/bin/python" ]; }

echo "== arm_agent setup =="
echo "   repo      : $ARM_AGENT_DIR"
echo "   root      : $ROOT"
echo "   conda     : $CONDA_ROOT"
echo "   model dir : $ARM_AGENT_MODEL_DIR"
echo "   check-only: $CHECK_ONLY"
echo

failed=0
report() {  # report <label> <ok:0|1> <detail>
    if [ "$2" = 0 ]; then printf '   [OK]   %-28s %s\n' "$1" "$3"
    else printf '   [MISS] %-28s %s\n' "$1" "$3"; failed=$((failed + 1)); fi
}

# ------------------------------------------------------------------ 1. gl_sw
if want gl_sw; then
    echo "== 1/4 gl_sw (software EGL) =="
    if env_exists "$ENV_GL" && [ "$CHECK_ONLY" = 0 ]; then
        echo "   already present, skipping (FORCE=1 to recreate)"
    elif [ "$CHECK_ONLY" = 0 ]; then
        FORCE=${FORCE:-0} bash "$HERE/setup_gl_sw_env.sh"
    fi
    if [ -e "$GL_LIB/libEGL.so.1" ]; then report "libEGL.so.1" 0 "$GL_LIB"
    else report "libEGL.so.1" 1 "run: bash scripts/setup_gl_sw_env.sh"; fi
    echo
fi

# ----------------------------------------------------------------- 2. libero
if want libero; then
    echo "== 2/4 libero (MuJoCo + robosuite) =="
    if env_exists "$ENV_SIM" && [ "$CHECK_ONLY" = 0 ]; then
        echo "   env already present, skipping"
    elif [ "$CHECK_ONLY" = 0 ]; then
        bash "$HERE/setup_sim_env.sh"
    fi

    # LIBERO itself: cloned into third_party/ (gitignored) and installed
    # editable. Pinned to the commit that was verified here so a future upstream
    # change cannot silently alter the task definitions.
    LIBERO_DIR=$ARM_AGENT_DIR/third_party/LIBERO
    LIBERO_COMMIT=${LIBERO_COMMIT:-8f1084e}
    if [ ! -d "$LIBERO_DIR/.git" ] && [ "$CHECK_ONLY" = 0 ]; then
        echo "   cloning LIBERO into third_party/ (commit $LIBERO_COMMIT)"
        mkdir -p "$ARM_AGENT_DIR/third_party"
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
        git -C "$LIBERO_DIR" checkout -q "$LIBERO_COMMIT"
        PYTHONNOUSERSITE=1 "$PY_SIM" -m pip install -e "$LIBERO_DIR"
    fi

    if env_exists "$ENV_SIM"; then
        if PYTHONNOUSERSITE=1 "$PY_SIM" -c "import libero, robosuite, mujoco" 2>/dev/null; then
            v=$("$PY_SIM" -c "import mujoco, robosuite; print(f'mujoco {mujoco.__version__}, robosuite {robosuite.__version__}')")
            report "sim env imports" 0 "$v"
        else
            # This is the failure that costs the most time to diagnose: without
            # the EGL library paths, `import mujoco` raises
            # "Cannot initialize a EGL device display", which reads like a broken
            # GPU stack and is really just a missing LD_LIBRARY_PATH.
            if LD_LIBRARY_PATH=$GL_LIB LIBGL_DRIVERS_PATH=$GL_DRI \
               MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
               PYTHONNOUSERSITE=1 "$PY_SIM" -c "import libero, robosuite, mujoco" 2>/dev/null; then
                report "sim env imports" 0 "OK with EGL env (must run via scripts/simenv.sh)"
            else
                report "sim env imports" 1 "run: bash scripts/setup_sim_env.sh"
            fi
        fi
    else
        report "sim env" 1 "run: bash scripts/setup_sim_env.sh"
    fi
    echo
fi

# ------------------------------------------------------------------ 3. agent
if want agent; then
    echo "== 3/4 agent (torch + transformers) =="
    if env_exists "$ENV_AGENT" && [ "$CHECK_ONLY" = 0 ]; then
        echo "   env already present, skipping"
    elif [ "$CHECK_ONLY" = 0 ]; then
        bash "$HERE/setup_agent_env.sh"
    fi
    if env_exists "$ENV_AGENT"; then
        if v=$(PYTHONNOUSERSITE=1 "$PY_AGENT" - <<'PY' 2>/dev/null
import importlib.metadata as md
print(" ".join(f"{p} {md.version(p)}" for p in ("torch", "transformers", "bitsandbytes")))
PY
); then report "agent env" 0 "$v"
        else report "agent env" 1 "run: bash scripts/setup_agent_env.sh"; fi
    else
        report "agent env" 1 "run: bash scripts/setup_agent_env.sh"
    fi
    echo
fi

# ------------------------------------------------------------------ 4. model
if want model; then
    echo "== 4/4 model weights =="
    if [ -f "$ARM_AGENT_MODEL_DIR/model.safetensors.index.json" ]; then
        report "weights present" 0 "$(du -sh "$ARM_AGENT_MODEL_DIR" 2>/dev/null | cut -f1) at $ARM_AGENT_MODEL_DIR"
    elif [ "$CHECK_ONLY" = 0 ]; then
        bash "$HERE/download_model.sh"
    else
        report "weights" 1 "run: bash scripts/download_model.sh"
    fi
    echo
fi

if [ "$failed" -gt 0 ]; then
    echo "== $failed item(s) need attention =="
    exit 1
fi

cat <<'EOF'
== all set ==

Smoke test (no GPU, ~2 minutes) -- this is the recommended first run:
    bash scripts/simenv.sh scripts/smoke_libero_e1.py
    bash scripts/simenv.sh scripts/probe_action_semantics.py
    bash scripts/simenv.sh scripts/probe_harness_dryrun.py

Real LLM episode (needs the GPU node):
    bash scripts/run_m1_on_gpu.sh 1
    python3 scripts/analyze_episode.py
EOF
