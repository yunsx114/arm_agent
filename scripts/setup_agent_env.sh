#!/usr/bin/env bash
# Create the `qwen35` env: torch + transformers + 4bit inference for the agent.
#
# VERSION CHOICES (for this cluster: rtx2080ti / gpu026, driver 570.133.20 = CUDA 12.8)
#   torch==2.8.0        -> the cu128 PyPI build; matches the driver and still
#                          ships sm_75 (Turing) kernels. Do NOT take a cu130
#                          torch: it requires driver >=580.
#   transformers==5.12.1 -> first version carrying the `qwen3_5` architecture.
#   NO flash-attn / causal-conv1d / fla: Turing is unsupported, and the model
#   falls back to plain PyTorch (it prints exactly that warning at startup).
#   bitsandbytes 0.50.2 -> NF4 4bit quantisation; peak 7.38GiB for the 9B model.
#
# WHY uv, NOT pip (measured):
#   pip downloads one file at a time and the mirrors throttle a single TCP
#   connection to 150-250 KB/s. This env is ~5GB (torch 888MB + cudnn 674MB +
#   cublas/nccl/cusolver...): pip ran >2h without finishing, and pip.conf's
#   extra-index-url=pypi.ngc.nvidia.com does not resolve inside the cluster, so
#   every package also waited through 5 retries. uv concurrently downloads
#   (measured 19 connections, 9MB/s) and finished in 22 minutes.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

PKGS=(
    "torch==2.8.0" "torchvision==0.23.0"
    "transformers==5.12.1"
    accelerate bitsandbytes safetensors pillow modelscope
)

echo "==> creating conda env '$ENV_AGENT' (python 3.11) in $CONDA_ROOT"
"$CONDA" create -y -n "$ENV_AGENT" python=3.11

if [ -x "$UV" ]; then
    echo "==> installing with uv (parallel, index: $PIP_INDEX)"
    "$UV" pip install --python "$PY_AGENT" --index-url "$PIP_INDEX" "${PKGS[@]}"
else
    echo "==> uv not found at $UV -- falling back to pip (much slower)" >&2
    # PIP_CONFIG_FILE=/dev/null skips the cluster's pip.conf, whose
    # extra-index-url cannot be resolved and costs 5 retries per package.
    PIP_CONFIG_FILE=/dev/null "$PY_AGENT" -m pip install --index-url "$PIP_INDEX" \
        --trusted-host mirrors.aliyun.com "${PKGS[@]}"
fi

echo
echo "==> version check"
"$PY_AGENT" - <<'PY'
import importlib.metadata as md
for p in ("torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "modelscope"):
    try:
        print(f"    {p:14s} {md.version(p)}")
    except Exception:
        print(f"    {p:14s} <MISSING>")
PY

cat <<EOF

==> done. python: $PY_AGENT

    Note: torch.cuda.get_arch_list() is [] on the login node (no CUDA context).
    On the GPU node it must contain 'sm_75'; if it does not, the 4bit kernels
    will fail at load time, not at import time.

    Next: bash scripts/download_model.sh
EOF
