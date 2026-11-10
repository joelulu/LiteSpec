#!/usr/bin/env bash
# Parallel, reusable VLM eligibility preprocessing; no Target weights loaded.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$REPO/.venv-vlm/bin/python"
    [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
fi
export PYTHON_BIN PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${PRECHECK_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl}"
export IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing VLM Python: $PYTHON_BIN" >&2; exit 1; }
[[ -s "$TRAIN_DATA" ]] || { echo "Missing/empty training data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "Missing local Target: $TARGET_MODEL" >&2; exit 1; }
if [[ -z "${CONFIG:-}" ]]; then
    CONFIG="$(mktemp --suffix=.yaml)"
    trap 'rm -f -- "$CONFIG"' EXIT
    "$PYTHON_BIN" tests/dflash_context_config.py "${DRAFT_CONTEXT:-full}" "$CONFIG"
fi
args=(
    --config "$CONFIG" --train-data "$TRAIN_DATA" --target-model "$TARGET_MODEL"
    --cache-root "${VALIDATION_CACHE_ROOT:-$DLITE_WORKSPACE/runs/vlm_validation}"
    --search-root "$DLITE_WORKSPACE/runs" --max-samples "${MAX_SAMPLES:-0}"
    --workers "${VALIDATION_WORKERS:-8}" --chunk-size "${VALIDATION_CHUNK_SIZE:-64}"
)
if [[ -n "${REUSE_VALIDATION_FROM:-}" ]]; then
    args+=(--reuse-validation-from "$REUSE_VALIDATION_FROM")
fi
"$PYTHON_BIN" tests/prepare_vlm_validation.py "${args[@]}" "$@"
