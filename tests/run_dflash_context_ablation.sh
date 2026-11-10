#!/usr/bin/env bash
# Train four variants on eight GPUs, then evaluate on one GPU.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/dflash_context_20k_1ep_$(date +%Y%m%d_%H%M%S)_$$}"
cd "$REPO"
unset RESULT_DIR
bash tests/train_dflash_context.sh
bash tests/evaluate_dflash_context.sh
