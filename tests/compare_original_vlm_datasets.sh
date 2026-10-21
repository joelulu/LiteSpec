#!/usr/bin/env bash
# AR vs original DLite on the user's local five vision datasets.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export ORIGINAL_DLITE="${ORIGINAL_DLITE:-$HOME/LMM/lmm_model/Qwen3.5-4B-DLite}"
export MODELS="ar dlite_original"
export BENCHMARK_MODE=static_batch
export BATCH_SIZES="${BATCH_SIZES:-1}"
export REFERENCE_INPUT=1
export SAMPLE_SEED="${SAMPLE_SEED:-42}"
export PROMPT_FIELD="${PROMPT_FIELD:-reference}"
DATA_DIR="${DATA_DIR:-$HOME/LMM/liyt/datasets/jsonl_datas}"
export IMAGE_ROOT="${IMAGE_ROOT:-$DATA_DIR/imgs}"
export EVAL_SAMPLES="${SAMPLES_PER_DATASET:-${EVAL_SAMPLES:-16}}"
export EVAL_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${EVAL_MAX_NEW_TOKENS:-512}}"
read -r -a datasets <<< "${DATASETS:-charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k}"
read -r -a modes <<< "${MODES:-text vision}"
((${#datasets[@]} && ${#modes[@]})) || { echo "Nonempty DATASETS and MODES required" >&2; exit 1; }
declare -A seen_datasets=() seen_modes=()
for dataset in "${datasets[@]}"; do
    case "$dataset" in charxiv_val|flickr30k|textvqa|hr_bench_4k|hr_bench_8k) ;; *) echo "Unknown dataset: $dataset" >&2; exit 1 ;; esac
    [[ ! ${seen_datasets[$dataset]+yes} ]] || { echo "Duplicate dataset: $dataset" >&2; exit 1; }
    seen_datasets[$dataset]=1
done
for mode in "${modes[@]}"; do
    case "$mode" in text|vision) ;; *) echo "Unknown mode: $mode (choose text vision)" >&2; exit 1 ;; esac
    [[ ! ${seen_modes[$mode]+yes} ]] || { echo "Duplicate mode: $mode" >&2; exit 1; }
    seen_modes[$mode]=1
done
GROUP_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/original_vlm_datasets_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$GROUP_DIR" ]] || { echo "Choose a fresh RESULT_DIR" >&2; exit 1; }
mkdir -p "$GROUP_DIR"
GROUP_DIR="$(cd -- "$GROUP_DIR" && pwd)"
cd "$REPO_ROOT"
# Inspects actual source records/images before loading any model. Never rewrites them.
"$PYTHON_BIN" tests/reference_vlm_data.py --data-dir "$DATA_DIR" --image-root "$IMAGE_ROOT" \
    --datasets "${datasets[@]}" --modes "${modes[@]}" --samples "$EVAL_SAMPLES" --seed "$SAMPLE_SEED" \
    --prompt-field "$PROMPT_FIELD" --output "$GROUP_DIR/inspection.json" 2>&1 | tee "$GROUP_DIR/inspection.log"
if [[ "${INSPECT_ONLY:-0}" == 1 ]]; then
    echo "Inspection: $GROUP_DIR/inspection.json"
    exit 0
fi
for dataset in "${datasets[@]}"; do
    for mode in "${modes[@]}"; do
        echo "=== $dataset / $mode / B=$BATCH_SIZES ==="
        EVAL_DATA="$DATA_DIR/$dataset.jsonl" EVAL_DATASET_NAME="$dataset" EVAL_SPLIT=unspecified \
        INPUT_MODE="$mode" RESULT_DIR="$GROUP_DIR/$dataset/$mode" \
            bash tests/compare_vlm_16.sh 2>&1 | tee "$GROUP_DIR/${dataset}_${mode}.log"
    done
done
"$PYTHON_BIN" tests/report_original_vlm_datasets.py --root "$GROUP_DIR" --datasets "${datasets[@]}" --modes "${modes[@]}"
echo "Tables: $GROUP_DIR/comparison.md"
echo "Metrics: $GROUP_DIR/comparison.csv"
