#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${RUN_GROUP:?Set RUN_GROUP to the completed context training directory}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}"
export RESULT_DIR="${RESULT_DIR:-$RUN_GROUP/eval_$(date +%Y%m%d_%H%M%S)_$$}"
exec "$PYTHON_BIN" tests/evaluate_dflash_context.py
