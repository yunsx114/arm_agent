#!/usr/bin/env bash
# Run the E6 VLM spatial-understanding probe on a GPU node.
#   bash scripts/run_e6_on_gpu.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

DIR=$ARM_AGENT_DIR
LOG="$DIR/outputs/logs/e6_spatial.log"
mkdir -p "$DIR/outputs/logs"

echo "==> srun on rtx2080ti: E6 spatial probe"
srun --partition=rtx2080ti --account=gpulab02 --qos=rtx2080ti \
     --nodes=1 --gres=gpu:1 --time=00:30:00 --job-name=e6_spatial \
     "$PY_AGENT" -u "$DIR/scripts/probe_vlm_spatial_e6.py" 2>&1 | tee "$LOG"
