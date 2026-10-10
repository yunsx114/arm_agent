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
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
PROMPT_MODE=${PROMPT_MODE:-full}
MODEL_DIR=${ARM_AGENT_MODEL_DIR}
mkdir -p "$IPC_DIR" "$LOG_DIR"

case "$EPISODES" in
    ''|*[!0-9]*) echo "episodes must be a non-negative integer: $EPISODES" >&2; exit 2 ;;
esac
if [ ! -d "$MODEL_DIR" ]; then
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
SIM_EGL_MODE=${SIM_EGL_MODE:-hardware} \
    bash "$HERE/simenv.sh" -u -m arm_agent.cli sim-server \
    --ipc-dir "$IPC_DIR" --tcp-port "$TCP_PORT" --idle-exit 7200 \
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

echo "==> running $EPISODES episode(s) with $LOAD model (log: $AGENT_LOG)"
set +e
ARM_AGENT_MODEL_DIR="$MODEL_DIR" \
    PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
    bash "$HERE/agentenv.sh" -u -m arm_agent.cli run \
    --ipc-dir "$IPC_DIR" --tcp-host 127.0.0.1 --tcp-port "$TCP_PORT" \
    --episodes "$EPISODES" --max-turns "$MAXTURNS" \
    --model-dir "$MODEL_DIR" --load "$LOAD" --gpu-mem-gib "$GPU_MEM_GIB" \
    --max-new-tokens "$MAX_NEW_TOKENS" --prompt-mode "$PROMPT_MODE" \
    "${BUDGET_ARGS[@]}" \
    "${RECORD_ARGS[@]}" 2>&1 | tee "$AGENT_LOG"
status=${PIPESTATUS[0]}
set -e

echo "==> stopping local simulator"
exit "$status"
