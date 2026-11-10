#!/usr/bin/env bash
# Full/Pivot1 on separate four-GPU groups, sharing data and initialization.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl}"
export ALGORITHM=dflash EPOCHS="${EPOCHS:-2}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-16}" IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}" REPORT_TO="${REPORT_TO:-tensorboard}"
export SAVE_TRAINING_STATE=0 TORCH_DISABLE_ADDR2LINE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PREPARE_VALIDATION="${PREPARE_VALIDATION:-1}" VALIDATION_WORKERS="${VALIDATION_WORKERS:-8}"
unset CONFIG OUTPUT_DIR MAX_STEPS RESUME

[[ -x "$PYTHON_BIN" ]] || { echo "Missing VLM Python: $PYTHON_BIN" >&2; exit 1; }
[[ -s "$TRAIN_DATA" ]] || { echo "Missing training data: $TRAIN_DATA" >&2; exit 1; }
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    [[ -s "$INIT_CHECKPOINT/config.json" ]] || { echo "Missing initial checkpoint: $INIT_CHECKPOINT" >&2; exit 1; }
fi
if [[ -n "${REUSE_VALIDATION_FROM:-}" ]]; then
    for file in effective_config.json skipped_training.jsonl; do
        [[ -f "$REUSE_VALIDATION_FROM/$file" ]] || { echo "Missing validation file: $REUSE_VALIDATION_FROM/$file" >&2; exit 1; }
    done
fi

RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/${RUN_PREFIX:-dflash_full_pivot1}_${EPOCHS}ep_mb${MICRO_BATCH_SIZE}_acc${ACCUMULATION_STEPS}_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$RUN_GROUP" ]] || { echo "Choose a fresh output directory: $RUN_GROUP" >&2; exit 1; }
mkdir -p "$RUN_GROUP"
for mode in full pivot1; do
    mkdir -p "$RUN_GROUP/$mode"
    "$PYTHON_BIN" tests/dflash_context_config.py "$mode" "$RUN_GROUP/$mode/config.yaml"
done

train_one() (
    export CUDA_VISIBLE_DEVICES="$1" NPROC_PER_NODE=4
    export OUTPUT_DIR="$RUN_GROUP/$2" CONFIG="$RUN_GROUP/$2/config.yaml"
    bash scripts/dlite/run_training_vlm.sh > "$OUTPUT_DIR/train.log" 2>&1
)
full_gpus="${FULL_GPUS:-0,1,2,3}"
pivot_gpus="${PIVOT_GPUS:-4,5,6,7}"
train_one "$full_gpus" full &
FULL_PID=$!
train_one "$pivot_gpus" pivot1 &
PIVOT_PID=$!
echo "Full: GPUs=$full_gpus; launcher PID=$FULL_PID"
echo "Pivot1: GPUs=$pivot_gpus; launcher PID=$PIVOT_PID"
echo "Each run: $EPOCHS epochs; global batch=$((4 * MICRO_BATCH_SIZE * ACCUMULATION_STEPS))"
echo "Data: $TRAIN_DATA; max_samples=$MAX_SAMPLES"
echo "Initialization: ${INIT_CHECKPOINT:-from scratch}; fresh optimizer/schedule"
echo "Validation: ${REUSE_VALIDATION_FROM:-automatic compatible-cache search}"
echo "Output: $RUN_GROUP"
echo "tail -f \"$RUN_GROUP/full/train.log\" \"$RUN_GROUP/pivot1/train.log\""
status=0
wait "$FULL_PID" || status=1
wait "$PIVOT_PID" || status=1
[[ "$status" == 0 ]] || { echo "A training run failed; inspect train.log" >&2; exit 1; }
echo "$RUN_GROUP/full/last"
echo "$RUN_GROUP/pivot1/last"
