#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
[[ -x "$PYTHON_BIN" ]] || { echo "Missing VLM Python: $PYTHON_BIN" >&2; exit 1; }
RESULT_DIR="${FULL_VISUAL_RESULT_DIR:-$DLITE_WORKSPACE/runs/full_visual_n32_$(date +%Y%m%d_%H%M%S)_$$}"
read -r -a gpus <<< "${VISUAL_EVAL_GPUS:-0 1 2 3 4 5 6 7}"
args=(--workspace "$DLITE_WORKSPACE" --output-dir "$RESULT_DIR" --gpus "${gpus[@]}"
      --scheduling "${VISUAL_EVAL_SCHEDULING:-condition}")
[[ -z "${SOURCE_EVAL_DIR:-}" ]] || args+=(--source-eval-dir "$SOURCE_EVAL_DIR")
exec "$PYTHON_BIN" tests/evaluate_full_visual.py "${args[@]}" "$@"
