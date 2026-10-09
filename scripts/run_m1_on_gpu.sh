#!/usr/bin/env bash
# Full two-node M1 run: sim server on the login node, LLM agent on gpu026.
#
#   bash scripts/run_m1_on_gpu.sh [episodes]
#
# Env knobs:
#   RECORD=1     -> pass --record: one mp4 per episode (cameras + model I/O)
#   MAXTURNS=N   -> per-episode turn budget (default 60)
#
# 60 is deliberate: a healthy run reaches the basket in ~20 turns, so 30 s of
# video is enough. If it has not placed the object by 60, something is wrong in
# the middle and more turns only burn GPU time (120-turn runs were measured at
# 1700-1900 s and still failed).
#
# Why two nodes (measured, DESIGN.md E3/E4): gpu026 cannot execute MuJoCo
# (CPU lacks SSE4.2 -> SIGILL) and cannot reach the login node over TCP
# (isolated subnet), so the two processes talk through the /lab NFS file-drop
# IPC instead. Everything below is the glue for that.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

DIR=$ARM_AGENT_DIR
EPISODES=${1:-1}
IPC_DIR="$DIR/runtime/ipc"
LOG_DIR="$DIR/outputs/logs"
mkdir -p "$LOG_DIR" "$IPC_DIR"

RECORD_FLAG=""
[ "${RECORD:-0}" = "1" ] && RECORD_FLAG="--record"
#   PROMPT_MODE=full|plain  -> system prompt variant (default full)
#     plain = P0a of the harness-stripping plan: task + frame + tool semantics,
#     with all scene-specific patches (heights, radii, 7-step recipe) removed.
MAXTURNS=${MAXTURNS:-60}
PROMPT_MODE=${PROMPT_MODE:-full}

# TCP transport (default) instead of the NFS file-drop.
# WHY (measured 2026-10-08): the NFS file-drop costs 30 s PER CALL because NFS
# `acdirmin` defaults to 30 s -- a cmd_*.json the client just wrote stays
# invisible to the server's directory glob for up to 30 s. Timing breakdown of
# one `move`: LLM 8.7 s, server-side work 0.56 s, client-side wait 30.0 s.
# login01 carries 192.168.82.239/23, the SAME /23 as gpu026 (192.168.82.46);
# that path answers in 0.67 ms. Set TCP_MODE=0 to use the old NFS path.
TCP_PORT=${TCP_PORT:-45678}
# Prefer the /23 that gpu026 lives on (192.168.82-83.x). login01 also has
# 192.168.81.8, which is a DIFFERENT subnet and would NOT be routable from the
# GPU node -- picking it would silently reintroduce the 30 s stall.
SVC_IP=$(hostname -I | tr ' ' '\n' | grep -E '^192\.168\.8[23]\.' | head -1)
if [ -z "$SVC_IP" ] || [ "${TCP_MODE:-1}" = "0" ]; then
  echo "==> transport: NFS file-drop (TCP disabled or no 192.168.x address)"
  SERVER_TCP_ARGS=""
  PING_TCP_ARGS=""
  RUN_TCP_ARGS=""
else
  echo "==> transport: TCP -> $SVC_IP:$TCP_PORT"
  SERVER_TCP_ARGS="--tcp-port $TCP_PORT"
  PING_TCP_ARGS="--tcp-host $SVC_IP --tcp-port $TCP_PORT"
  RUN_TCP_ARGS="--tcp-host $SVC_IP --tcp-port $TCP_PORT"
fi

# Stale commands/responses from a previous run would be replayed as garbage.
rm -f "$IPC_DIR"/cmd_*.json "$IPC_DIR"/resp_*.json

echo "==> starting sim server on login node (log: $LOG_DIR/m1_server.log)"
setsid nohup env PYTHONNOUSERSITE=1 \
  PYTHONPATH="$DIR/src" \
  LD_LIBRARY_PATH="$GL_LIB" \
  LIBGL_DRIVERS_PATH="$GL_DRI" \
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  "$PY_SIM" -u -m arm_agent.cli sim-server \
  --ipc-dir "$IPC_DIR" --idle-exit 1800 $SERVER_TCP_ARGS \
  > "$LOG_DIR/m1_server.log" 2>&1 &
echo "    server pid $!"

echo "==> waiting for the server to answer pings ..."
for _ in $(seq 1 60); do
  if env PYTHONNOUSERSITE=1 PYTHONPATH="$DIR/src" \
     "$PY_SIM" -m arm_agent.cli ipc-ping \
     --ipc-dir "$IPC_DIR" $PING_TCP_ARGS > /dev/null 2>&1; then
    echo "    server alive"
    break
  fi
  sleep 2
done

echo "==> submitting LLM episode(s): $EPISODES on rtx2080ti"
# expandable_segments: the measured OOM was a 266 MiB request failing with
# ~1.2 GiB free — i.e. allocator fragmentation, not raw exhaustion.
# 2h walltime: ~40 s/turn measured on this 4bit stack.
srun --partition=rtx2080ti --account=gpulab02 --qos=rtx2080ti \
     --nodes=1 --gres=gpu:1 --time=02:00:00 --job-name=m1_episode \
     env PYTHONPATH="$DIR/src" TOKENIZERS_PARALLELISM=false \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     ARM_AGENT_MODEL_DIR="$ARM_AGENT_MODEL_DIR" \
     "$PY_AGENT" -u -m arm_agent.cli run \
     --ipc-dir "$IPC_DIR" --episodes "$EPISODES" --max-turns "$MAXTURNS" $RECORD_FLAG \
     --prompt-mode "$PROMPT_MODE" \
     $RUN_TCP_ARGS \
     2>&1 | tee "$LOG_DIR/m1_episode.log"

echo "==> stopping sim server"
env PYTHONNOUSERSITE=1 PYTHONPATH="$DIR/src" IPC_DIR="$IPC_DIR" \
  SVC_IP="$SVC_IP" TCP_PORT="$TCP_PORT" RUN_TCP_ARGS="$RUN_TCP_ARGS" \
  "$PY_SIM" - > /dev/null 2>&1 <<'PYEOF' || true
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
from arm_agent.sim.client import SimClient
args = os.environ.get("RUN_TCP_ARGS", "").split()
ep = (args[1], int(args[3])) if args else None
SimClient(os.environ["IPC_DIR"], timeout_s=10, endpoint=ep).close()
PYEOF
echo "==> done. Transcripts: $DIR/outputs/episodes/"
