#!/usr/bin/env bash
# Four supplied checkpoints + AR, same 16 prompts, C=1/4/8 on one GPU.
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
DSPARK_3EP="${DSPARK_3EP:-$THREE_RUN/dspark/last}"
checkpoints=(--checkpoint dlite_6ep "$DLITE_6EP"
             --checkpoint dlite_3ep "$DLITE_3EP"
             --checkpoint dflash_3ep "$DFLASH_3EP"
             --checkpoint dspark_3ep "$DSPARK_3EP")
if [[ ${MODELS+x} ]]; then
    read -r -a models <<< "${MODELS//,/ }"
    ((${#models[@]})) || { echo "MODELS must not be empty; use ar for target only" >&2; exit 1; }
    checkpoints=()
    declare -A seen_models=()
    for model in "${models[@]}"; do
        [[ ! ${seen_models[$model]+yes} ]] || { echo "Duplicate model: $model" >&2; exit 1; }
        seen_models[$model]=1
        case "$model" in
            ar) ;;
            dlite_original) checkpoints+=(--checkpoint "$model" "${ORIGINAL_DLITE:-$HOME/LMM/lmm_model/Qwen3.5-4B-DLite}") ;;
            dlite_6ep) checkpoints+=(--checkpoint "$model" "$DLITE_6EP") ;;
            dlite_3ep) checkpoints+=(--checkpoint "$model" "$DLITE_3EP") ;;
            dflash_3ep) checkpoints+=(--checkpoint "$model" "$DFLASH_3EP") ;;
            dflash_text) checkpoints+=(--checkpoint "$model" "${DFLASH_TEXT:?Set DFLASH_TEXT to the text-trained checkpoint}") ;;
            dflash_vlm) checkpoints+=(--checkpoint "$model" "${DFLASH_VLM:?Set DFLASH_VLM to the multimodal-trained checkpoint}") ;;
            dspark_3ep) checkpoints+=(--checkpoint "$model" "$DSPARK_3EP") ;;
            *) echo "Unknown model: $model" >&2; exit 1 ;;
        esac
    done
    echo "Selected models: ${models[*]}; AR baseline is always included"
elif [[ "${ONLY_ORIGINAL:-0}" == 1 ]]; then
    checkpoints=(--checkpoint dlite_original "${ORIGINAL_DLITE:-$HOME/LMM/lmm_model/Qwen3.5-4B-DLite}")
elif [[ -n "${ORIGINAL_DLITE:-}" ]]; then
    checkpoints+=(--checkpoint dlite_original "$ORIGINAL_DLITE")
fi
RESULT_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/vlm_compare_$(date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
read -r -a concurrency <<< "${CONCURRENCIES:-1 4 8}"
extra_args=()
if [[ "${REPORT_RAW_SPEEDUP:-0}" == 1 ]]; then
    extra_args+=(--report-raw-speedup)
fi
cd "$REPO_ROOT"
if [[ "${BENCHMARK_MODE:-reference}" == static_batch ]]; then
    read -r -a batches <<< "${BATCH_SIZES:-1 8}"
    input_args=()
    if [[ "${REFERENCE_INPUT:-0}" == 1 ]]; then
        input_args+=(--reference-input --input-mode "${INPUT_MODE:-vision}" --seed "${SAMPLE_SEED:-42}"
                     --prompt-field "${PROMPT_FIELD:-reference}")
        [[ -z "${IMAGE_ROOT:-}" ]] || input_args+=(--image-root "$IMAGE_ROOT")
    fi
    [[ -z "${IMAGE_MAX_PIXELS:-}" ]] || input_args+=(--image-max-pixels "$IMAGE_MAX_PIXELS")
    [[ -z "${IMAGE_MIN_PIXELS:-}" ]] || input_args+=(--image-min-pixels "$IMAGE_MIN_PIXELS")
    if [[ "${PROFILE_DECODE:-0}" == 1 ]]; then
        input_args+=(--profile-decode --profile-samples "${PROFILE_SAMPLES:-4}")
        [[ "${PROFILE_KERNELS:-0}" != 1 ]] || input_args+=(--profile-kernels)
    fi
    exec "$PYTHON_BIN" tests/benchmark_vlm_static_batch.py \
        --target-model "$TARGET_MODEL" --data "$EVAL_DATA" --data-split "${EVAL_SPLIT:-unspecified}" \
        --dataset-name "${EVAL_DATASET_NAME:-unspecified}" \
        "${checkpoints[@]}" --batch-sizes "${batches[@]}" \
        --max-samples "${EVAL_SAMPLES:-16}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-128}" \
        --warmup "${WARMUP:-1}" --attention-backend "${ATTENTION_BACKEND:-auto}" \
        --output-dir "$RESULT_DIR" "${input_args[@]}"
fi
exec "$PYTHON_BIN" tests/benchmark_vlm_checkpoints.py \
    --target-model "$TARGET_MODEL" --data "$EVAL_DATA" --data-split "${EVAL_SPLIT:-unspecified}" \
    --dataset-name "${EVAL_DATASET_NAME:-unspecified}" \
    "${checkpoints[@]}" \
    --max-samples "${EVAL_SAMPLES:-16}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-128}" \
    --concurrency "${concurrency[@]}" --warmup "${WARMUP:-1}" \
    --verification "${VERIFICATION:-block}" --output-dir "$RESULT_DIR" "${extra_args[@]}"
