#!/usr/bin/env bash
# Create the `qwen35` env: torch + transformers + 4bit inference for the agent.
#
# VERSION CHOICES (validated on RTX 5090 / torch 2.8.0+cu128)
#   torch==2.8.0        -> the cu128 PyPI build; matches the driver and still
#                          ships sm_120 kernels.
#   transformers==5.12.1 -> first version carrying the `qwen3_5` architecture.
#   fla-core + causal-conv1d -> Qwen3.5 linear-attention fast path. The
#                          causal-conv1d wheel is selected from the upstream
#                          release assets to match torch, CUDA major, Python,
#                          and the C++ ABI.
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
    "fla-core==0.5.2" "flash-linear-attention==0.5.2" "einops==0.8.1"
    # Episode video (RECORD=1). imageio-ffmpeg SHIPS its own ffmpeg binary, so
    # no system ffmpeg is required; without these two the recorder still writes
    # every PNG frame but finalize() cannot encode the mp4.
    imageio imageio-ffmpeg
)

echo "==> creating conda env '$ENV_AGENT' (python 3.11) in $CONDA_ROOT"
if [ -x "$PY_AGENT" ]; then
    echo "    environment already exists; installing/updating packages in place"
else
    "$CONDA" create -y -n "$ENV_AGENT" python=3.11
fi

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

install_causal_conv1d() {
    local version=${CAUSAL_CONV1D_VERSION:-1.7.0}
    local wheel_url
    local wheel_file

    if "$PY_AGENT" -c 'import causal_conv1d' >/dev/null 2>&1; then
        echo "==> causal-conv1d is already installed"
        return
    fi

    read -r wheel_url wheel_file <<EOF
$("$PY_AGENT" - "$version" <<'PY'
import sys
import torch

version = sys.argv[1]
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
cuda = torch.version.cuda
if not cuda:
    raise SystemExit("PyTorch has no CUDA runtime; cannot select causal-conv1d wheel")
cuda_tag = f"cu{cuda.split('.')[0]}"
torch_tag = "torch" + ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi_tag = "TRUE" if torch.compiled_with_cxx11_abi() else "FALSE"
wheel = (
    f"causal_conv1d-{version}+{cuda_tag}{torch_tag}"
    f"cxx11abi{abi_tag}-{python_tag}-{python_tag}-linux_x86_64.whl"
)
url = (
    f"https://github.com/Dao-AILab/causal-conv1d/releases/download/"
    f"v{version}/{wheel.replace('+', '%2B')}"
)
print(url, wheel)
PY
)"
EOF

    if [ -z "$wheel_url" ] || [ -z "$wheel_file" ]; then
        echo "ERROR: could not determine a causal-conv1d wheel for this environment" >&2
        return 1
    fi
    echo "==> installing $wheel_file"
    if [ -x "$UV" ]; then
        "$UV" pip install --python "$PY_AGENT" --no-deps "$wheel_url"
    else
        PIP_CONFIG_FILE=/dev/null "$PY_AGENT" -m pip install --no-deps "$wheel_url"
    fi
    "$PY_AGENT" -c 'import causal_conv1d; print("    causal-conv1d import: ok")'
}

install_causal_conv1d

echo
echo "==> version check"
"$PY_AGENT" - <<'PY'
import importlib.metadata as md
for p in (
    "torch", "torchvision", "transformers", "accelerate", "bitsandbytes",
    "modelscope", "fla-core", "flash-linear-attention", "causal-conv1d",
):
    try:
        print(f"    {p:14s} {md.version(p)}")
    except Exception:
        print(f"    {p:14s} <MISSING>")
PY

cat <<EOF

==> done. python: $PY_AGENT

    Note: torch.cuda.get_arch_list() is [] on a CPU-only node. On the GPU node,
    verify that it contains the architecture required by the installed GPU.

    Next: bash scripts/download_model.sh
EOF
