#!/usr/bin/env bash
# Run the LIBERO simulator and Qwen3.5 agent on one machine.
#
#   bash scripts/run_single_node.sh [episodes]
#
# Useful overrides:
#   ARM_AGENT_MODEL_DIR=/data/models/Qwen3.5-9B
#   LOAD=4bit|bf16|fp16       GPU_MEM_GIB=29
#   MAXTURNS=60 MAX_NEW_TOKENS=160 PROMPT_MODE=full|plain
#   RECORD=1 SIM_EGL_MODE=hardware TCP_PORT=45679
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

EPISODES=${1:-1}
IPC_DIR=${IPC_DIR:-$ARM_AGENT_DIR/runtime/ipc-single-node}
LOG_DIR=${LOG_DIR:-$ARM_AGENT_DIR/outputs/logs}
TCP_PORT=${TCP_PORT:-45679}
MAXTURNS=${MAXTURNS:-60}
LOAD=${LOAD:-4bit}
GPU_MEM_GIB=${GPU_MEM_GIB:-29}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-384}
PROMPT_MODE=${PROMPT_MODE:-full}
MODEL_DIR=${ARM_AGENT_MODEL_DIR}
mkdir -p "$IPC_DIR" "$LOG_DIR"

case "$EPISODES" in
    ''|*[!0-9]*) echo "episodes must be a non-negative integer: $EPISODES" >&2; exit 2 ;;
esac
# The in-process 9B needs its weights on disk; the Strata backend does not read
# them at all (the engine on 127.0.0.1:8080 owns the model), and requiring the
# folder there would only force a pointless 18 GB copy.
if [ "${BACKEND:-local}" != "strata" ] && [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model directory does not exist: $MODEL_DIR" >&2
    exit 2
fi

SERVER_LOG="$LOG_DIR/single_node_server.log"
AGENT_LOG="$LOG_DIR/single_node_agent.log"
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Do not replay commands from a previous interrupted run.
rm -f "$IPC_DIR"/cmd_*.json "$IPC_DIR"/resp_*.json

echo "==> starting local simulator on 127.0.0.1:$TCP_PORT"
# SEED changes the sim's randomization; TASK_ID/INIT_STATE pick a different scene
# layout (LIBERO ships 10 init states per task). Both were unreachable before,
# which made every run replay the SAME scene -- and in the 12th run the basket got
# shoved out of the camera's view by the wandering arm, after which no release
# position could ever be correct. Being able to vary the scene is a precondition
# for telling "the model cannot aim" apart from "the target moved".
SCENE_ARGS=(--seed "${SEED:-0}" --task-id "${TASK_ID:-0}")
# Render size. 档 B measured that the MODEL's pointing error in the 256 px
# third-person view is >12 cm while the can is only ~9 px wide there (the `[aim]`
# line in the server log reports the gap to the true object per call), so the
# resolution is a variable of the experiment, not a constant.
SCENE_ARGS+=(--cam "${CAM:-256}")
# 档 B: `truth` lets the harness resolve object names through the simulator's own
# positions; `pixels` removes that path entirely (aim/target on pixels) and annotates
# every frame with a coordinate grid + the gripper crosshair. The server ENFORCES
# it, and the harness mirrors it so the advertised tool surface matches.
SCENE_ARGS+=(--observation-mode "${OBS_MODE:-truth}" --pixel-grid "${PIXEL_GRID:-32}")
# Radius of the `aim` ring, in pixels. An experiment variable on purpose: 3 px was
# the first proposal and is too small to judge a gap against (see
# pixel_grounding.AIM_MARK_RADIUS_PX for the measurement).
SCENE_ARGS+=(--aim-mark "${AIM_MARK_PX:-6}")
SIM_EGL_MODE=${SIM_EGL_MODE:-hardware} \
    bash "$HERE/simenv.sh" -u -m arm_agent.cli sim-server \
    --ipc-dir "$IPC_DIR" --tcp-port "$TCP_PORT" --idle-exit 7200 \
    "${SCENE_ARGS[@]}" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "==> waiting for simulator"
for _ in $(seq 1 120); do
    if bash "$HERE/agentenv.sh" -m arm_agent.cli ipc-ping \
        --ipc-dir "$IPC_DIR" --tcp-host 127.0.0.1 --tcp-port "$TCP_PORT" \
        >/dev/null 2>&1; then
        echo "    simulator is alive (pid $SERVER_PID)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: simulator exited; see $SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done

if ! bash "$HERE/agentenv.sh" -m arm_agent.cli ipc-ping \
    --ipc-dir "$IPC_DIR" --tcp-host 127.0.0.1 --tcp-port "$TCP_PORT" \
    >/dev/null 2>&1; then
    echo "ERROR: simulator did not answer within 120 seconds; see $SERVER_LOG" >&2
    exit 1
fi

RECORD_ARGS=()
[ "${RECORD:-0}" = "1" ] && RECORD_ARGS+=(--record)
# Optional budget overrides. Leave unset to use harness defaults
# (MAX_HISTORY_MESSAGES=32 on a 32 GiB card; 4 is the 10 GiB tier).
BUDGET_ARGS=()
[ -n "${HISTORY_MESSAGES:-}" ] && BUDGET_ARGS+=(--history-messages "$HISTORY_MESSAGES")
[ -n "${IMAGE_KEEP:-}" ] && BUDGET_ARGS+=(--image-keep "$IMAGE_KEEP")
# Released-gate knobs removed 2026-10-11: the two-role verifier that had to
# approve every carrying `gripper open` is gone (16/16 approvals, 0 blocks, and
# an audit showed all 16 release poses were correct -- it re-checked a solved
# problem for ~4 % of wall clock). The harness still LOGS `_gate_arithmetic` at
# every release; nothing blocks an action any more.
GATE_ARGS=()
# Which model answers. LOCAL (default) = the in-process Qwen3.5-9B, which needs
# the GPU to itself; STRATA = HTTP to the Strata server on 127.0.0.1:8080, whose
# engine already holds ~31 GiB of VRAM -- so loading the 9B as well would OOM.
BACKEND_ARGS=(--backend "${BACKEND:-local}")
[ -n "${STRATA_URL:-}" ] && BACKEND_ARGS+=(--strata-url "$STRATA_URL")
# The run side must agree with the server's observation mode (see SCENE_ARGS).
[ -n "${OBS_MODE:-}" ] && BACKEND_ARGS+=(--observation-mode "$OBS_MODE")
[ -n "${REASONING_EFFORT:-}" ] && BACKEND_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
[ -n "${THINKING_EFFORT:-}" ] && BACKEND_ARGS+=(--thinking-effort "$THINKING_EFFORT")

echo "==> running $EPISODES episode(s) with $LOAD model (log: $AGENT_LOG)"
set +e
ARM_AGENT_MODEL_DIR="$MODEL_DIR" \
    PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
    bash "$HERE/agentenv.sh" -u -m arm_agent.cli run \
    --ipc-dir "$IPC_DIR" --tcp-host 127.0.0.1 --tcp-port "$TCP_PORT" \
    --episodes "$EPISODES" --max-turns "$MAXTURNS" --init-state "${INIT_STATE:-0}" \
    --model-dir "$MODEL_DIR" --load "$LOAD" --gpu-mem-gib "$GPU_MEM_GIB" \
    --max-new-tokens "$MAX_NEW_TOKENS" --prompt-mode "$PROMPT_MODE" \
    "${BUDGET_ARGS[@]}" \
    "${GATE_ARGS[@]}" \
    "${BACKEND_ARGS[@]}" \
    "${RECORD_ARGS[@]}" 2>&1 | tee "$AGENT_LOG"
status=${PIPESTATUS[0]}
set -e

echo "==> stopping local simulator"
exit "$status"
