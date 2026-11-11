#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
read -r -a gpus <<< "${LONGCTX_GPUS:-0 1 2 3 4 5 6 7}"
args=(--workspace "$DLITE_WORKSPACE" --data-dir "${LONGCTX_DATA_DIR:-$DLITE_WORKSPACE/data/longctx_mathvista100}"
    --output-dir "${LONGCTX_RESULT_DIR:-$DLITE_WORKSPACE/runs/longctx_full_pivots_n16_$(date +%Y%m%d_%H%M%S)_$$}"
    --samples 16 --lengths 0 8192 32768 65536 --repeats 2 --gpus "${gpus[@]}"
    --image-max-tokens 1024 --max-new-tokens 256 --seed 42 --profile-samples 2)
[[ -z "${SOURCE_EVAL_DIR:-}" ]] || args+=(--source-eval-dir "$SOURCE_EVAL_DIR")
exec "$PYTHON_BIN" tests/evaluate_dflash_longctx.py "${args[@]}" "$@"
