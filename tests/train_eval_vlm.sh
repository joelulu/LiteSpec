#!/usr/bin/env bash
# Copy to dlite-vlm/train_eval_vlm.sh, or run directly from LiteSpec/tests.
# bash train_eval_vlm.sh [all|train|eval]
set -euo pipefail
MODE="${1:-all}"
case "$MODE" in all|train|eval) ;; *) echo "Usage: bash $0 [all|train|eval]" >&2; exit 2 ;; esac
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$HERE/LiteSpec" ]]; then
    DEFAULT_WORKSPACE="$HERE"
else
    DEFAULT_WORKSPACE="$(cd -- "$HERE/../.." && pwd)"
fi
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$DEFAULT_WORKSPACE}"
REPO_ROOT="$DLITE_WORKSPACE/LiteSpec"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/val.jsonl}"
if [[ "$MODE" == eval && -z "${OUTPUT_DIR:-}" && -z "${CHECKPOINT:-}" ]]; then
    echo 'For eval, set OUTPUT_DIR to the completed run, or CHECKPOINT to its last directory.' >&2
    exit 2
fi
if [[ -z "${OUTPUT_DIR:-}" && -n "${CHECKPOINT:-}" ]]; then
    export OUTPUT_DIR="$(dirname -- "$CHECKPOINT")"
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    export OUTPUT_DIR="$DLITE_WORKSPACE/runs/dlite_vlm_aligned_6ep_$(date +%Y%m%d_%H%M%S)"
fi
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/last}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -s "$EVAL_DATA" || "$MODE" == train ]] || { echo "Missing validation manifest: $EVAL_DATA" >&2; exit 1; }
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
if [[ "$MODE" != eval ]]; then
    # This wrapper always requests the full aligned recipe, without smoke caps.
    unset MAX_STEPS MAX_SAMPLES
    export CONFIG="$REPO_ROOT/scripts/config/vlm/sft_student_aligned.yaml"
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
    export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    echo "Training output: $OUTPUT_DIR"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
fi
[[ "$MODE" != train ]] || exit 0
[[ -s "$CHECKPOINT/config.json" ]] || { echo "Missing completed checkpoint: $CHECKPOINT" >&2; exit 1; }
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$OUTPUT_DIR/eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$EVAL_OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
status=0
# Block mode checks actual speculative verification. Sequential mode is a
# correctness/acceptance diagnostic only, not an inference speed benchmark.
for verification in block sequential; do
    echo "Evaluation: $verification; checkpoint: $CHECKPOINT"
    if "$PYTHON_BIN" tests/run_vlm_eval.py \
        --target-model "$TARGET_MODEL" --draft-model "$CHECKPOINT" \
        --data "$EVAL_DATA" --device cuda \
        --max-samples "${EVAL_SAMPLES:-32}" \
        --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-128}" --warmup 1 \
        --verification "$verification" \
        --output "$EVAL_OUTPUT_DIR/$verification.jsonl" \
        2>&1 | tee "$EVAL_OUTPUT_DIR/$verification.log"; then
        echo "$verification evaluation passed"
    else
        echo "$verification evaluation failed; inspect $EVAL_OUTPUT_DIR/$verification.log" >&2
        status=1
    fi
done
echo "Evaluation results: $EVAL_OUTPUT_DIR"
exit "$status"
