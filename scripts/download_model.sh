#!/usr/bin/env bash
# Download the Qwen3.5-9B weights (about 19GB) from ModelScope.
#
#   bash scripts/download_model.sh
#   DEST=/data/models/Qwen3.5-9B bash scripts/download_model.sh
#   MODEL_ID=Qwen/Qwen3.5-4B bash scripts/download_model.sh
#
# Resumable: snapshot_download skips complete files, so re-running after an
# interruption continues where it stopped.
#
# WHY ModelScope: the cluster cannot reach huggingface.co, and ModelScope is the
# domestic mirror of the same weights (identical file set, including the
# preprocessor_config.json the vision path needs).
#
# WHY THE PARALLEL KNOBS ARE NOT OPTIONAL (measured):
#   The ModelScope SDK disables parallel downloading by default
#   (MODELSCOPE_DOWNLOAD_PARALLELS=1), which pulls each large shard over a
#   SINGLE TCP connection. Once the CDN throttles that one connection to
#   ~160KB/s -- and it does -- a 5.3GB shard never finishes (measured: 8h+ for
#   the full 19GB). Setting it to 16 switches to chunked parallel download
#   (160MB chunks, up to 16 connections); the same download then completes in
#   ~1h20m. The threshold must also be lowered, or files below 1GB (the
#   default threshold) still take the single-connection path.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-9B}
DEST=${DEST:-$ARM_AGENT_MODEL_DIR}

# Find a python that can import modelscope: the agent env by default, then
# whatever is on PATH.
PY=""
for c in "$PY_AGENT" "$(command -v python3)"; do
    if [ -x "$c" ] && "$c" -c "import modelscope" 2>/dev/null; then
        PY=$c
        break
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: no python with modelscope found." >&2
    echo "       Run scripts/setup_agent_env.sh first, or: pip install modelscope" >&2
    exit 1
fi

export MODELSCOPE_DOWNLOAD_PARALLELS=${MODELSCOPE_DOWNLOAD_PARALLELS:-16}
export MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB=${MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB:-100}

mkdir -p "$(dirname "$DEST")"
echo "==> python   : $PY"
echo "==> model    : $MODEL_ID"
echo "==> dest     : $DEST"
echo "==> parallel : $MODELSCOPE_DOWNLOAD_PARALLELS connections (threshold ${MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB}MB)"

MODEL_ID="$MODEL_ID" DEST="$DEST" "$PY" - <<'PY'
import os
from modelscope import snapshot_download

path = snapshot_download(os.environ["MODEL_ID"], local_dir=os.environ["DEST"])
print("==> downloaded:", path)
PY

echo
echo "==> verifying the files the agent actually opens"
MODEL_DIR="$DEST" "$PY" - <<'PY'
import json
import os
from pathlib import Path

d = Path(os.environ["MODEL_DIR"])
required = ["config.json", "tokenizer_config.json", "preprocessor_config.json"]
index = d / "model.safetensors.index.json"
missing = [f for f in required if not (d / f).exists()]
if not index.exists():
    missing.append("model.safetensors.index.json")
if missing:
    raise SystemExit(f"    MISSING: {missing}")

shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
total = 0
for s in shards:
    p = d / s
    if not p.exists():
        raise SystemExit(f"    MISSING SHARD: {s}")
    total += p.stat().st_size

print(f"    OK  {len(shards)} shards, {total / 2**30:.1f} GiB")
print(f"    OK  config.json / tokenizer_config.json / preprocessor_config.json")
print(f"    model dir = {d}")
PY

cat <<EOF

==> done.

    Run the agent with:
      bash scripts/run_m1_on_gpu.sh 1

    scripts/cli.py defaults to '\$ARM_AGENT_ROOT/qwen35_demo/models/Qwen3.5-9B'
    (ARM_AGENT_MODEL_DIR). If you downloaded elsewhere, point it there:
      ARM_AGENT_MODEL_DIR=$DEST bash scripts/run_m1_on_gpu.sh 1
EOF
