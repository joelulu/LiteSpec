#!/usr/bin/env bash
# Preserve completed Pivot5/B exports; retry C in a fresh directory, then eval all three.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
FAILED_RUN_GROUP="${1:-${FAILED_RUN_GROUP:-}}"
[[ -n "$FAILED_RUN_GROUP" ]] || { echo 'Set FAILED_RUN_GROUP to the failed experiment-2 group' >&2; exit 2; }
export RUN_GROUP="$(cd -- "$FAILED_RUN_GROUP" && pwd -P)"
for mode in pivot5 pivot5_B; do
    [[ -f "$RUN_GROUP/$mode/last/config.json" ]] || { echo "Missing completed $mode checkpoint: $RUN_GROUP/$mode/last" >&2; exit 1; }
done
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3}" NPROC_PER_NODE=4
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
[[ "$MICRO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'MICRO_BATCH_SIZE must be positive' >&2; exit 2; }
(( 256 % (4 * MICRO_BATCH_SIZE) == 0 )) || { echo 'Micro batch must divide global batch 256' >&2; exit 2; }
export ACCUMULATION_STEPS=$((256 / (4 * MICRO_BATCH_SIZE)))
export ALGORITHM=dflash EPOCHS=5 MAX_SAMPLES=0 IMAGE_MAX_TOKENS=1024
export SAVE_EVERY_EPOCH=1 SAVE_TRAINING_STATE=0 PREPARE_VALIDATION=1
export VALIDATION_WORKERS="${VALIDATION_WORKERS:-8}" REPORT_TO="${REPORT_TO:-tensorboard}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DISABLE_ADDR2LINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRAIN_DATA="$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl"
export REUSE_VALIDATION_FROM="$DLITE_WORKSPACE/runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb"
unset INIT_CHECKPOINT RESUME MAX_STEPS CONFIG OUTPUT_DIR
[[ -x "$PYTHON_BIN" && -s "$TRAIN_DATA" && -d "$TARGET_MODEL" ]] || { echo 'Missing Python, data or Target' >&2; exit 1; }
for file in effective_config.json skipped_training.jsonl; do
    [[ -f "$REUSE_VALIDATION_FROM/$file" ]] || { echo "Missing preflight: $file" >&2; exit 1; }
done
"$PYTHON_BIN" tests/check_sw_token_source.py
"$PYTHON_BIN" - <<'PY'
import os, torch
mask = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(mask) == len(set(mask)) == 4 and torch.cuda.device_count() == 4, 'Need four distinct visible GPUs'
PY
export OUTPUT_DIR="${C_RETRY_DIR:-$RUN_GROUP/pivot5_C_retry_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Choose fresh C_RETRY_DIR: $OUTPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
export CONFIG="$OUTPUT_DIR/config.yaml"
"$PYTHON_BIN" tests/dflash_context_config.py pivot5_C "$CONFIG"
"$PYTHON_BIN" - <<'PY'
import json, os, subprocess
from pathlib import Path
root = Path(os.environ['RUN_GROUP'])
out = Path(os.environ['OUTPUT_DIR'])
manifest = dict(initialization='from_scratch', retrained=['pivot5_C'],
    reused={mode: str(root/mode/'last') for mode in ('pivot5','pivot5_B')},
    settings={k: os.environ[k] for k in ('TRAIN_DATA','REUSE_VALIDATION_FROM','EPOCHS',
        'CUDA_VISIBLE_DEVICES','NPROC_PER_NODE','MICRO_BATCH_SIZE','ACCUMULATION_STEPS')},
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
(out/'retry_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f'Retrying C only, from scratch: {out}', flush=True)
PY
"$PYTHON_BIN" tests/check_dflash_layer_adapters.py --device cuda --flex \
    2>&1 | tee "$OUTPUT_DIR/layer_adapter_check.log"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    tests/check_dflash_layer_adapters.py --fsdp 2>&1 | tee "$OUTPUT_DIR/layer_adapter_fsdp_check.log"
bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
[[ -f "$OUTPUT_DIR/last/config.json" && -d "$OUTPUT_DIR/epoch_5" ]] || { echo 'Missing final C epoch export' >&2; exit 1; }
export EVAL_GPUS="${CUDA_VISIBLE_DEVICES//,/ }"
export EVAL_SAMPLES=32 VIDEO_SAMPLES=32 EVAL_BATCH_SIZE=1 EVAL_MAX_NEW_TOKENS=256 SAMPLE_SEED=42
export EVAL_PROMPT_FIELD=reference VIDEO_FRAMES=16 VIDEO_MAX_PIXELS_PER_FRAME=200704
export RESULT_DIR="$OUTPUT_DIR/eval_all_b1_n32"
bash tests/evaluate_dflash_all.sh \
    --checkpoint pivot5 "$RUN_GROUP/pivot5/last" \
    --checkpoint pivot5_B "$RUN_GROUP/pivot5_B/last" \
    --checkpoint pivot5_C "$OUTPUT_DIR/last" 2>&1 | tee "$OUTPUT_DIR/evaluate.log"
echo "C retry and three-model evaluation complete: $RESULT_DIR/comparison.md"
