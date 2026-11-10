#!/usr/bin/env bash
# Eight distinct GPUs, one dataset per GPU. Existing checkpoints are read only.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
: "${RUN_GROUP:?Set RUN_GROUP to the intended Full/Pivot1 training directory}"
RESULT_DIR="${RESULT_DIR:-$RUN_GROUP/eval_all_$(date +%Y%m%d_%H%M%S)_$$}"
read -r -a gpus <<< "${EVAL_GPUS:-0 1 2 3 4 5 6 7}"
args=(
    --run-group "$RUN_GROUP" --output-dir "$RESULT_DIR" --workspace "$DLITE_WORKSPACE"
    --target-model "${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
    --data-dir "${DATA_DIR:-$HOME/LMM/liyt/datasets/jsonl_datas}"
    --video-data-root "${VIDEO_DATA_ROOT:-$HOME/LMM/liyt/datasets/eval_bench}"
    --gpus "${gpus[@]}" --samples "${EVAL_SAMPLES:-32}" --video-samples "${VIDEO_SAMPLES:-16}"
    --batch-size "${EVAL_BATCH_SIZE:-1}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}"
    --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}" --seed "${SAMPLE_SEED:-42}" --warmup 1
    --frames "${VIDEO_FRAMES:-16}" --max-pixels-per-frame "${VIDEO_MAX_PIXELS_PER_FRAME:-200704}"
    --prompt-field "${EVAL_PROMPT_FIELD:-reference}"
)
[[ -z "${IMAGE_ROOT:-}" ]] || args+=(--image-root "$IMAGE_ROOT")
[[ -z "${TEXT_DATA:-}" ]] || args+=(--text-data "$TEXT_DATA")
exec "$PYTHON_BIN" tests/evaluate_dflash_all.py "${args[@]}" "$@"
