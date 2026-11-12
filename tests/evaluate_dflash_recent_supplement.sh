#!/usr/bin/env bash
# Repeat evaluation independently after all three new runs complete.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}"
: "${RUN_GROUP:?Set RUN_GROUP to the completed recent supplement directory}"
"$PYTHON_BIN" tests/prepare_dflash_recent_supplement.py finalize --run-group "$RUN_GROUP"
source "$RUN_GROUP/launch.env"
checkpoints=()
for mode in full recent256 recent128 recent64 pivot1 pivot5 pivot12 recent0; do
    case "$mode" in
        recent256|recent64|recent0) checkpoint="$RUN_GROUP/$mode/epoch_5" ;;
        *) checkpoint="$BASELINE_RUN_GROUP/$mode/epoch_5" ;;
    esac
    checkpoints+=(--checkpoint "$mode" "$checkpoint")
done
export EVAL_SAMPLES=32 VIDEO_SAMPLES=32 EVAL_BATCH_SIZE=1 EVAL_MAX_NEW_TOKENS=256 SAMPLE_SEED=42
export IMAGE_MAX_TOKENS=1024 VIDEO_FRAMES=16 VIDEO_MAX_PIXELS_PER_FRAME=200704
export EVAL_REUSE_SELECTION_FROM="$BASELINE_EVAL_DIR"
export RESULT_DIR="${RECENT_EVAL_RESULT_DIR:-$RUN_GROUP/eval_all_b1_n32_$(date +%Y%m%d_%H%M%S)_$$}"
status=0
bash tests/evaluate_dflash_all.sh "${checkpoints[@]}" 2>&1 | tee "$RUN_GROUP/evaluate_all.log" || status=1
export SOURCE_EVAL_DIR="$BASELINE_EVAL_DIR"
export LONGCTX_GPUS="${LONGCTX_GPUS:-0 1 2 3}"
export LONGCTX_RESULT_DIR="${RECENT_LONGCTX_RESULT_DIR:-$RUN_GROUP/eval_longctx_b1_n32_r1_$(date +%Y%m%d_%H%M%S)_$$}"
bash tests/evaluate_dflash_longctx.sh --samples 32 --repeats 1 \
    --supplement-run-group "$RUN_GROUP" \
    --models full recent256 recent128 recent64 pivot1 pivot5 pivot12 recent0 \
    2>&1 | tee "$RUN_GROUP/evaluate_longctx.log" || status=1
echo "All-dataset table: $RESULT_DIR/comparison.md"
echo "Long-context table: $LONGCTX_RESULT_DIR/longctx_comparison.md"
exit "$status"
