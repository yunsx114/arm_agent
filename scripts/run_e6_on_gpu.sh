#!/usr/bin/env bash
# Run the E6 VLM spatial-understanding probe on a GPU node.
#   bash scripts/run_e6_on_gpu.sh
set -euo pipefail

ROOT=/lab/haoq_lab/cse12311731
DIR="$ROOT/arm_agent"
PY="$ROOT/miniconda3/envs/qwen35/bin/python"
LOG="$DIR/outputs/logs/e6_spatial.log"
mkdir -p "$DIR/outputs/logs"

echo "==> srun on rtx2080ti: E6 spatial probe"
srun --partition=rtx2080ti --account=gpulab02 --qos=rtx2080ti \
     --nodes=1 --gres=gpu:1 --time=00:30:00 --job-name=e6_spatial \
     "$PY" -u "$DIR/scripts/probe_vlm_spatial_e6.py" 2>&1 | tee "$LOG"
