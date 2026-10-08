#!/usr/bin/env bash
# Run the E5 tool-calling probe on a GPU node (see probe_tool_calling_e5.py).
# Cluster facts: only gpu026 (rtx2080ti partition) is usable; it has no AVX so
# only the LLM runs there, never MuJoCo (see DESIGN.md E3).
#
#   bash scripts/run_e5_on_gpu.sh
#
# Logs to outputs/logs/e5_tool_calling.log on the shared /lab NFS.
set -euo pipefail

ROOT=/lab/haoq_lab/cse12311731
DIR="$ROOT/arm_agent"
PY="$ROOT/miniconda3/envs/qwen35/bin/python"
LOG="$DIR/outputs/logs/e5_tool_calling.log"
mkdir -p "$DIR/outputs/logs"

echo "==> srun on rtx2080ti: E5 tool-calling probe"
srun --partition=rtx2080ti --account=gpulab02 --qos=rtx2080ti \
     --nodes=1 --gres=gpu:1 --time=00:40:00 --job-name=e5_toolcall \
     "$PY" -u "$DIR/scripts/probe_tool_calling_e5.py" 2>&1 | tee "$LOG"
