#!/usr/bin/env bash
# Four matched from-scratch runs on GPUs 0-3; regeneration owns its servers.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export CLIENT_PYTHON="$PYTHON_BIN"
export PYTHONPATH="$REPO:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export DATA_ROOT="${DATA_ROOT:-$DLITE_WORKSPACE/data/finevision_qwen35_30w_img1024_b8b16}"
export COMMON_DATA_DIR="$DATA_ROOT/common"
export CUDA_VISIBLE_DEVICES=0,1,2,3 DATA_GPUS=0,1,2,3 NPROC_PER_NODE=4
export EVAL_GPUS="0 1 2 3" LONGCTX_GPUS="0 1 2 3"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 TORCH_DISABLE_ADDR2LINE=1
export VALIDATION_WORKERS="${VALIDATION_WORKERS:-8}"
stage="${1:-all}"
case "$stage" in all|data|train|eval) ;; *) echo 'Usage: run_dflash_30w.sh all|data|train|eval' >&2; exit 2 ;; esac
[[ -x "$PYTHON_BIN" && -d "$TARGET_MODEL" ]] || { echo 'Missing VLM Python or Target model' >&2; exit 1; }
unset INIT_CHECKPOINT RESUME MAX_STEPS REUSE_VALIDATION_FROM CONFIG OUTPUT_DIR PROMPT_MANIFEST
export ALGORITHM=dflash EPOCHS=5 MICRO_BATCH_SIZE=2 ACCUMULATION_STEPS=32
export MAX_SAMPLES=0 IMAGE_MAX_TOKENS=1024 SAVE_EVERY_EPOCH=1 SAVE_TRAINING_STATE=0
export PREPARE_VALIDATION=1 PRECHECK_CUDA_VISIBLE_DEVICES=0
export REPORT_TO="${REPORT_TO:-tensorboard}"
export LONGCTX_DATA_DIR="${LONGCTX_DATA_DIR:-$DLITE_WORKSPACE/data/longctx_mathvista100}"

if [[ "$stage" == all || "$stage" == data ]]; then
    mkdir -p "$DATA_ROOT"
    if [[ -d "$COMMON_DATA_DIR" ]]; then
        "$PYTHON_BIN" tests/dflash_30w.py verify --common "$COMMON_DATA_DIR" --target-model "$TARGET_MODEL"
        echo "Reusing completed common preprocessing: $COMMON_DATA_DIR"
    else
        export SOURCE_DATA="${SOURCE_DATA:-$HOME/LMM/liyt/datasets/FineVision/FineVision_subset/finevision_sharegpt.jsonl}"
        [[ -s "$SOURCE_DATA" ]] || { echo "Missing FineVision source: $SOURCE_DATA" >&2; exit 1; }
        export MAX_SAMPLES=300000 MAX_NEW_TOKENS=256 MIN_ANSWER_TOKENS=1
        export MAX_LENGTH=4096 CONTEXT_LENGTH=4096 ADAPTIVE_ANSWER_BUDGET=1
        export PREPARE_OUTPUT="$DATA_ROOT/regenerated"
        export CONCURRENCY="${CONCURRENCY:-8}"
        # Existing pipeline supports any number of replicas selected by DATA_GPUS.
        bash scripts/data/generate_finevision_8gpu.sh 2>&1 | tee "$DATA_ROOT/regenerate.log"
        "$PYTHON_BIN" tests/dflash_30w.py data --generated "$PREPARE_OUTPUT" \
            --common "$COMMON_DATA_DIR" --target-model "$TARGET_MODEL" \
            --workers "$VALIDATION_WORKERS" 2>&1 | tee "$DATA_ROOT/preflight.log"
    fi
    [[ "$stage" != data ]] || { echo "Data ready: $COMMON_DATA_DIR"; exit 0; }
fi
export MAX_SAMPLES=0
mkdir -p "$DATA_ROOT"
"$PYTHON_BIN" tests/dflash_30w.py inputs --workspace "$DLITE_WORKSPACE" \
    --longctx-data-dir "$LONGCTX_DATA_DIR" --result-file "$DATA_ROOT/evaluation_source.env"
source "$DATA_ROOT/evaluation_source.env"
if [[ "$stage" != eval ]]; then
    export RUN_GROUP="${SUITE_RUN_GROUP:-$DLITE_WORKSPACE/runs/dflash_full_pivot1_30w_5ep_b8_b16_mb2_acc32_$(date +%Y%m%d_%H%M%S)_$$}"
    "$PYTHON_BIN" tests/dflash_30w.py run --common "$COMMON_DATA_DIR" \
        --target-model "$TARGET_MODEL" --run-group "$RUN_GROUP"
    source "$RUN_GROUP/launch.env"
    "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.device_count() == 4, 'Need GPUs 0,1,2,3 visible'
PY
    # Real CPU-prefix parity/reload and CUDA/Flex/FSDP optimizer checks.
    "$PYTHON_BIN" tests/check_dflash_blocks.py --device cuda --flex \
        2>&1 | tee "$RUN_GROUP/block_check.log"
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
        tests/check_dflash_blocks.py --fsdp 2>&1 | tee "$RUN_GROUP/block_fsdp_check.log"
    for label in full_b8 pivot1_b8 full_b16 pivot1_b16; do
        block="${label##*_b}"
        export CONFIG="$RUN_GROUP/$label/config.yaml" OUTPUT_DIR="$RUN_GROUP/$label"
        export REUSE_VALIDATION_FROM="$COMMON_DATA_DIR/validation_b$block"
        echo "Training $label: GPUs 0-3, 5 epochs, global batch=4x2x32=256; $OUTPUT_DIR"
        bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    done
else
    : "${RUN_GROUP:?Set RUN_GROUP to the completed four-model experiment for eval}"
fi
"$PYTHON_BIN" tests/dflash_30w.py finalize --run-group "$RUN_GROUP"
source "$RUN_GROUP/launch.env"
checkpoints=()
for label in full_b8 pivot1_b8 full_b16 pivot1_b16; do
    checkpoints+=(--checkpoint "$label" "$RUN_GROUP/$label/epoch_5")
done
export EVAL_SAMPLES=32 VIDEO_SAMPLES=32 EVAL_BATCH_SIZE=1 EVAL_MAX_NEW_TOKENS=256 SAMPLE_SEED=42
export EVAL_PROMPT_FIELD=reference VIDEO_FRAMES=16 VIDEO_MAX_PIXELS_PER_FRAME=200704
# Input readiness selected and checked an earlier frozen selection when available.
export RESULT_DIR="$RUN_GROUP/eval_all_b1_n32_$(date +%Y%m%d_%H%M%S)_$$"
status=0
bash tests/evaluate_dflash_all.sh "${checkpoints[@]}" 2>&1 | tee "$RUN_GROUP/evaluate_all.log" || status=1
export LONGCTX_RESULT_DIR="$RUN_GROUP/eval_longctx_b1_n32_r1_$(date +%Y%m%d_%H%M%S)_$$"
bash tests/evaluate_dflash_longctx.sh --samples 32 --repeats 1 --suite-run-group "$RUN_GROUP" \
    2>&1 | tee "$RUN_GROUP/evaluate_longctx.log" || status=1
echo "Training group: $RUN_GROUP"
echo "All datasets: $RESULT_DIR/comparison.md"
echo "Long context: $LONGCTX_RESULT_DIR/longctx_comparison.md"
exit "$status"
