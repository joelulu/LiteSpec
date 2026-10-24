#!/usr/bin/env bash
# First collect EXACTLY 20k usable rows on CPU, then four-arm training + evaluation.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
SOURCE_TRAIN_DATA="${SOURCE_TRAIN_DATA:-${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl}}"
export IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export EPOCHS="${EPOCHS:-3}"
SUBSET_DIR="${SUBSET_DIR:-$DLITE_WORKSPACE/data/finevision_qwen35_gdn_valid20k_img${IMAGE_MAX_TOKENS}_b${BLOCK_SIZE}_s42}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
echo "Collecting exactly 20000 usable records from: $SOURCE_TRAIN_DATA"
echo "CPU preparation only; target weights are not loaded. Output: $SUBSET_DIR"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_BIN" tests/prepare_gdn_subset.py \
    --source "$SOURCE_TRAIN_DATA" --output-dir "$SUBSET_DIR" --target-model "$TARGET_MODEL" \
    --count 20000 --seed 42 --max-length "$MAX_LENGTH" --block-size "$BLOCK_SIZE" \
    --image-max-tokens "$IMAGE_MAX_TOKENS" --workers "${PREPARE_WORKERS:-4}"
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
    echo "Prepared exactly 20000 usable records: $SUBSET_DIR/train.jsonl"
    exit 0
fi
export TRAIN_DATA="$SUBSET_DIR/train.jsonl"
export PREPARED_VALIDATION_FROM="$SUBSET_DIR"
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/vlm_gdn_valid20k_${EPOCHS}ep_$(date +%Y%m%d_%H%M%S)_$$}"
echo "Preparation succeeded. Starting four-arm training on physical GPUs 4,5,6,7."
exec bash tests/train_gdn_ablation.sh
