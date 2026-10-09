#!/usr/bin/env bash
# Run the architectural-levers probe on the GPU node.
#   bash scripts/run_arch_levers_on_gpu.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

DIR=$ARM_AGENT_DIR
LOG="$DIR/outputs/logs/arch_levers.log"
mkdir -p "$DIR/outputs/logs"

echo "==> srun on rtx2080ti: architectural levers"
srun --partition=rtx2080ti --account=gpulab02 --qos=rtx2080ti \
     --nodes=1 --gres=gpu:1 --time=01:00:00 --job-name=arch_levers \
     env PYTHONPATH="$DIR/src" TOKENIZERS_PARALLELISM=false \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     ARM_AGENT_MODEL_DIR="$ARM_AGENT_MODEL_DIR" \
     "$PY_AGENT" -u "$DIR/scripts/probe_arch_levers.py" 2>&1 | tee "$LOG"
