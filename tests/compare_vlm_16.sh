#!/usr/bin/env bash
# Three supplied checkpoints + AR, same 16 prompts, C=1/4/8 on one GPU.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$HERE/LiteSpec" ]]; then
    DEFAULT_WORKSPACE="$HERE"
else
    DEFAULT_WORKSPACE="$(cd -- "$HERE/../.." && pwd)"
fi
DLITE_WORKSPACE="${DLITE_WORKSPACE:-$DEFAULT_WORKSPACE}"
REPO_ROOT="$DLITE_WORKSPACE/LiteSpec"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
EVAL_DATA="${EVAL_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/val.jsonl}"
THREE_RUN="$DLITE_WORKSPACE/runs/vlm_three_3ep_20261021_045040_2983971"
DLITE_6EP="${DLITE_6EP:-$DLITE_WORKSPACE/runs/dlite_vlm_aligned_6ep_20261021_022447/last}"
DLITE_3EP="${DLITE_3EP:-$THREE_RUN/dlite/last}"
DFLASH_3EP="${DFLASH_3EP:-$THREE_RUN/dflash/last}"
RESULT_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/vlm_compare_$(date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
read -r -a concurrency <<< "${CONCURRENCIES:-1 4 8}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" tests/benchmark_vlm_checkpoints.py \
    --target-model "$TARGET_MODEL" --data "$EVAL_DATA" \
    --checkpoint dlite_6ep "$DLITE_6EP" \
    --checkpoint dlite_3ep "$DLITE_3EP" \
    --checkpoint dflash_3ep "$DFLASH_3EP" \
    --max-samples "${EVAL_SAMPLES:-16}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-128}" \
    --concurrency "${concurrency[@]}" --warmup "${WARMUP:-1}" \
    --verification "${VERIFICATION:-block}" --output-dir "$RESULT_DIR"
