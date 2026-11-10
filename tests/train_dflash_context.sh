#!/usr/bin/env bash
# Five from-scratch DFlash runs; final weights only.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}" SAVE_TRAINING_STATE=0 ALGORITHM=dflash
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}" ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"
export EPOCHS="${EPOCHS:-1}" MAX_SAMPLES="${TRAIN_SAMPLES:-20000}" IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
export REPORT_TO="${REPORT_TO:-tensorboard}" TORCH_DISABLE_ADDR2LINE=1
export PREPARE_VALIDATION="${PREPARE_VALIDATION:-1}" VALIDATION_WORKERS="${VALIDATION_WORKERS:-8}"
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/dflash_context_20k_1ep_$(date +%Y%m%d_%H%M%S)_$$}"
unset INIT_CHECKPOINT RESUME MAX_STEPS CONFIG OUTPUT_DIR
[[ ! -e "$RUN_GROUP" ]] || { echo "Choose fresh RUN_GROUP: $RUN_GROUP" >&2; exit 1; }
"$PYTHON_BIN" tests/check_sw_token_source.py
read -r -a modes <<< "${CONTEXTS:-full recent128 pivot1 pivot5 pivot12}"
"$PYTHON_BIN" - "${modes[@]}" <<'PY'
import sys, os, torch
from litespec.vlm.dflash_context import CONTEXTS
assert sys.argv[1:] and len(set(sys.argv[1:])) == len(sys.argv[1:])
assert all(x in CONTEXTS for x in sys.argv[1:])
assert torch.cuda.device_count() == int(os.environ['NPROC_PER_NODE'])
PY
mkdir -p "$RUN_GROUP"
for context in "${modes[@]}"; do
    export OUTPUT_DIR="$RUN_GROUP/$context" CONFIG="$RUN_GROUP/$context/config.yaml"
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" tests/dflash_context_config.py "$context" "$CONFIG"
    echo "Training $context: $EPOCHS epochs; micro=$MICRO_BATCH_SIZE; accumulation=$ACCUMULATION_STEPS; GPUs=$NPROC_PER_NODE"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    [[ -f "$OUTPUT_DIR/last/config.json" ]] || exit 1
    export REUSE_VALIDATION_FROM="$OUTPUT_DIR"
done
echo "Final checkpoints: $RUN_GROUP/{full,recent128,pivot1,pivot5,pivot12}/last"
