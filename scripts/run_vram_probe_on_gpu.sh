#!/usr/bin/env bash
# Run the VRAM-vs-context probe on the GPU node.
#
#   bash scripts/run_vram_probe_on_gpu.sh
#   TARGETS=2000,4000,8000,16000 bash scripts/run_vram_probe_on_gpu.sh
#   IMAGE_SIDE=256 bash scripts/run_vram_probe_on_gpu.sh
#
# Long: the upper sweep points cost a quadratic prefill, and the sweep runs until
# it OOMs (that OOM is the measurement, not a failure). Budget 30-45 minutes and
# a 2h srun.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

DIR=$ARM_AGENT_DIR
LOG="$DIR/outputs/logs/vram_vs_context.log"
mkdir -p "$DIR/outputs/logs"

echo "==> srun on rtx2080ti: VRAM vs context length"
srun --partition=rtx2080ti --account=gpulab02 --qos=rtx2080ti \
     --nodes=1 --gres=gpu:1 --time=02:00:00 --job-name=vram_probe \
     env PYTHONPATH="$DIR/src" TOKENIZERS_PARALLELISM=false \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     ARM_AGENT_MODEL_DIR="$ARM_AGENT_MODEL_DIR" \
     IMAGE_SIDE="${IMAGE_SIDE:-160}" \
     TARGETS="${TARGETS:-2500,5000,8000,12000,16000,22000,30000,40000,55000,70000}" \
     "$PY_AGENT" -u "$DIR/scripts/probe_vram_vs_context.py" 2>&1 | tee "$LOG"
