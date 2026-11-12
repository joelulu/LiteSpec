#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export RESULT_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/longctx_four_n16_$(date +%Y%m%d_%H%M%S)_$$}"
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/dflash_exp1_10w_5ep_mb4_acc8_20261110_154705_2794634}"
read -r -a GPU_IDS <<< "${EVAL_GPUS:-0 1 2 3}"
echo "Results: $RESULT_DIR"
echo "Physical GPUs: ${GPU_IDS[*]}; models: AR Full Pivot1; B=1"
exec "$PYTHON_BIN" tests/evaluate_longctx_four.py \
 --workspace "$DLITE_WORKSPACE" \
 --data-dir "${LONGCTX_SUITE_DIR:-$DLITE_WORKSPACE/data/multimodal_longctx100}" \
 --output-dir "$RESULT_DIR" \
 --target-model "${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}" \
 --run-group "$RUN_GROUP" \
 --checkpoint-name "${CHECKPOINT_NAME:-epoch_5}" \
 --gpus "${GPU_IDS[@]}" \
 --samples "${EVAL_SAMPLES:-16}" \
 --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}" \
 --smoke-samples "${SMOKE_SAMPLES:-1}" "$@"
