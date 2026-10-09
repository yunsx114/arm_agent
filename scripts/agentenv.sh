#!/usr/bin/env bash
# Run a command inside the `qwen35` agent env with the repo on PYTHONPATH.
#
#   bash scripts/agentenv.sh -m arm_agent.cli run --episodes 1
#   bash scripts/agentenv.sh scripts/probe_harness_dryrun.py
#
# Unlike simenv.sh this needs no EGL: the agent process never renders. It must
# run on gpu026 (the only model-capable node) via srun; the login-node variant
# is only useful for text-only dry runs.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/_common.sh
source "$HERE/_common.sh"

export_pythonpath
export TOKENIZERS_PARALLELISM=false

exec "$PY_AGENT" "$@"
