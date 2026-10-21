#!/usr/bin/env bash
# Image/text direct SFT. Use the HF VLM environment, not the SGLang server env.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(dirname -- "$REPO_ROOT")"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
CONFIG="${CONFIG:-$REPO_ROOT/scripts/config/vlm/sft_student_aligned.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$WORKSPACE/data/finevision_qwen35_sglang_20000/train.jsonl}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/runs/dlite_vlm_aligned_6ep_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
[[ -s "$TRAIN_DATA" ]] || { echo "Missing/empty training data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "Missing local target model: $TARGET_MODEL" >&2; exit 1; }
echo "Python: $PYTHON_BIN"
echo "Data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
"$PYTHON_BIN" - "$NPROC_PER_NODE" <<'PY'
import os
import sys
import torch

try:
    workers = int(sys.argv[1])
except ValueError:
    sys.exit("NPROC_PER_NODE must be a positive integer")
if workers < 1:
    sys.exit("NPROC_PER_NODE must be a positive integer")
visible = torch.cuda.device_count()
mask = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
print(f"CUDA_VISIBLE_DEVICES={mask}; visible GPUs={visible}; workers={workers}", flush=True)
if visible < workers:
    sys.exit(
        f"Cannot launch {workers} workers with only {visible} visible GPUs. "
        "For eight-GPU training, set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
        "and NPROC_PER_NODE=8; otherwise choose an available GPU mask and matching worker count."
    )
PY
mkdir -p "$OUTPUT_DIR"
limits=()
if [[ "${SAVE_TRAINING_STATE:-0}" == 1 ]]; then
    limits+=(--save-training-state)
fi
if [[ -n "${REPORT_TO:-}" ]]; then
    limits+=(--report-to "$REPORT_TO")
fi
# No implicit 1000-step cap. Only explicit debug requests may stop early.
if [[ -n "${MAX_STEPS:-}" ]]; then
    limits+=(--max-steps "$MAX_STEPS")
    echo "WARNING: explicit MAX_STEPS=$MAX_STEPS debug cap is active"
fi
exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
    -m litespec.vlm.train_aligned --config "$CONFIG" \
    --target-model "$TARGET_MODEL" --train-data "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --max-samples "${MAX_SAMPLES:-0}" "${limits[@]}" "$@"
