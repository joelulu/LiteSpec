#!/usr/bin/env bash
# Image/text direct SFT. Use the HF VLM environment, not the SGLang server env.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(dirname -- "$REPO_ROOT")"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
CONFIG="${CONFIG:-$REPO_ROOT/scripts/config/vlm/sft_student_ddp.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$WORKSPACE/data/finevision_qwen35_sglang_20000/train.jsonl}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/runs/dlite_vlm_sft_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
[[ -s "$TRAIN_DATA" ]] || { echo "Missing/empty training data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "Missing local target model: $TARGET_MODEL" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
echo "Python: $PYTHON_BIN"
echo "Data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE:-8}" \
    -m litespec.vlm.train --config "$CONFIG" \
    --target-model "$TARGET_MODEL" --train-data "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" --max-steps "${MAX_STEPS:-1000}" \
    --max-samples "${MAX_SAMPLES:-0}" "$@"
