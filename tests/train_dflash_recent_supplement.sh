#!/usr/bin/env bash
# Three fresh 8-GPU runs; original experiment-1 exports remain read only.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}"
export BASELINE_RUN_GROUP="${BASELINE_RUN_GROUP:-$DLITE_WORKSPACE/runs/dflash_exp1_10w_5ep_mb4_acc8_20261110_154705_2794634}"
export RUN_GROUP="${RECENT_RUN_GROUP:-$DLITE_WORKSPACE/runs/dflash_recent_supplement_10w_5ep_mb4_acc8_$(date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 TORCH_DISABLE_ADDR2LINE=1
export PREPARE_VALIDATION=1 VALIDATION_WORKERS="${VALIDATION_WORKERS:-8}"
unset INIT_CHECKPOINT RESUME MAX_STEPS CONFIG OUTPUT_DIR
[[ -x "$PYTHON_BIN" ]] || { echo "Missing VLM interpreter: $PYTHON_BIN" >&2; exit 1; }
args=(prepare --baseline-run-group "$BASELINE_RUN_GROUP" --output-dir "$RUN_GROUP"
    --reuse-validation-from "${REUSE_VALIDATION_FROM:-$DLITE_WORKSPACE/runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb}")
[[ -z "${SOURCE_EVAL_DIR:-}" ]] || args+=(--source-eval-dir "$SOURCE_EVAL_DIR")
"$PYTHON_BIN" tests/prepare_dflash_recent_supplement.py "${args[@]}"
source "$RUN_GROUP/launch.env"
"$PYTHON_BIN" - <<'PY'
import os,torch
mask=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(mask)==len(set(mask))==torch.cuda.device_count()==8, 'Need exactly eight distinct, visible GPUs'
print('Training: recent256, recent64, recent0; each 8 GPUs, 5 epochs, batch=8x4x8=256',flush=True)
PY
[[ -f "${LONGCTX_DATA_DIR:-$DLITE_WORKSPACE/data/longctx_mathvista100}/manifest.json" ]] || {
    echo 'Prepare the existing 100-record MathVista/GovReport data before this suite.' >&2; exit 1;
}
# Real numerical / compiled Flex and production distributed optimizer checks.
"$PYTHON_BIN" tests/check_dflash_recent.py --device cuda --flex 2>&1 | tee "$RUN_GROUP/recent_check.log"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    tests/check_dflash_recent.py --fsdp 2>&1 | tee "$RUN_GROUP/recent_fsdp_check.log"
for mode in recent256 recent64 recent0; do
    export OUTPUT_DIR="$RUN_GROUP/$mode" CONFIG="$RUN_GROUP/$mode/config.yaml"
    echo "Training $mode: $OUTPUT_DIR"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    [[ -f "$OUTPUT_DIR/epoch_5/config.json" && -f "$OUTPUT_DIR/last/config.json" ]] || {
        echo "Missing completed epoch-5/last export: $mode" >&2; exit 1;
    }
done
"$PYTHON_BIN" tests/prepare_dflash_recent_supplement.py finalize --run-group "$RUN_GROUP"
export EVAL_GPUS="${CUDA_VISIBLE_DEVICES//,/ }"
bash tests/evaluate_dflash_recent_supplement.sh
echo "Supplement and eight-model evaluation complete: $RUN_GROUP"
