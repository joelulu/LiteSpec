#!/usr/bin/env bash
# Sequential training, per-epoch exports, then all eight benchmark datasets.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
case "${1:-}" in
    long)
        suite=exp1_10w_5ep; export EPOCHS=5 NPROC_PER_NODE=8 MAX_SAMPLES=0
        modes=(full recent128 pivot1 pivot5 pivot12)
        export TRAIN_DATA="$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl"
        export REUSE_VALIDATION_FROM="$DLITE_WORKSPACE/runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb"
        export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
        ;;
    adapters)
        suite=exp2_BC_10w_5ep; export EPOCHS=5 NPROC_PER_NODE=4 MAX_SAMPLES=0
        modes=(pivot5_B pivot5_C)
        export TRAIN_DATA="$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl"
        export REUSE_VALIDATION_FROM="$DLITE_WORKSPACE/runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb"
        export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3}"
        ;;
    *) echo "Usage: bash tests/train_dflash_research_suite.sh long|adapters" >&2; exit 2 ;;
esac
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
[[ "$MICRO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'MICRO_BATCH_SIZE must be positive' >&2; exit 2; }
(( 256 % (NPROC_PER_NODE * MICRO_BATCH_SIZE) == 0 )) || { echo 'Micro batch must divide global batch 256' >&2; exit 2; }
export ACCUMULATION_STEPS=$((256 / (NPROC_PER_NODE * MICRO_BATCH_SIZE)))
export IMAGE_MAX_TOKENS=1024 ALGORITHM=dflash SAVE_TRAINING_STATE=0 SAVE_EVERY_EPOCH=1
export PREPARE_VALIDATION=1 VALIDATION_WORKERS="${VALIDATION_WORKERS:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 TORCH_DISABLE_ADDR2LINE=1 REPORT_TO="${REPORT_TO:-tensorboard}"
export RUN_GROUP="${SUITE_RUN_GROUP:-$DLITE_WORKSPACE/runs/dflash_${suite}_mb${MICRO_BATCH_SIZE}_acc${ACCUMULATION_STEPS}_$(date +%Y%m%d_%H%M%S)_$$}"
unset INIT_CHECKPOINT RESUME MAX_STEPS CONFIG OUTPUT_DIR
[[ -x "$PYTHON_BIN" && -s "$TRAIN_DATA" && -d "$TARGET_MODEL" ]] || { echo 'Missing Python, data or Target model' >&2; exit 1; }
[[ ! -e "$RUN_GROUP" ]] || { echo "Choose a fresh SUITE_RUN_GROUP: $RUN_GROUP" >&2; exit 1; }
for file in effective_config.json skipped_training.jsonl; do
    [[ -f "$REUSE_VALIDATION_FROM/$file" ]] || { echo "Missing preflight: $REUSE_VALIDATION_FROM/$file" >&2; exit 1; }
done
"$PYTHON_BIN" tests/check_sw_token_source.py
"$PYTHON_BIN" - <<'PY'
import os, torch
mask = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
count = int(os.environ['NPROC_PER_NODE'])
assert len(mask) == len(set(mask)) == count, f'Need exactly {count} distinct GPU selectors'
assert torch.cuda.device_count() == count, 'Visible CUDA count differs from GPU selectors'
print('CUDA GPUs:', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
PY
mkdir -p "$RUN_GROUP"
for mode in "${modes[@]}"; do
    mkdir -p "$RUN_GROUP/$mode"
    "$PYTHON_BIN" tests/dflash_context_config.py "$mode" "$RUN_GROUP/$mode/config.yaml"
done
"$PYTHON_BIN" - "$suite" "${modes[@]}" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
keys = ('TRAIN_DATA','REUSE_VALIDATION_FROM','TARGET_MODEL','CUDA_VISIBLE_DEVICES',
        'NPROC_PER_NODE','EPOCHS','MICRO_BATCH_SIZE','ACCUMULATION_STEPS','MAX_SAMPLES','IMAGE_MAX_TOKENS')
root = Path(os.environ['RUN_GROUP'])
manifest = dict(suite=sys.argv[1], modes=sys.argv[2:], initialization='from_scratch',
                settings={key: os.environ[key] for key in keys}, global_batch=256,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                evaluation=dict(batch_size=1, samples=32, video_samples=32, seed=42, max_new_tokens=256,
                                checkpoints='last', datasets=['gsm8k','charxiv_val','textvqa','flickr30k',
                                'hr_bench_4k','hr_bench_8k','VideoDetailCaption','Video-MME']))
(root/'experiment.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
print('Output:', root, flush=True)
PY
# Check C's learnable Flex score modifier and optimizer ownership before any
# long training. No Target weights or external dataset are loaded by this test.
if [[ "$1" == adapters ]]; then
    "$PYTHON_BIN" tests/check_dflash_layer_adapters.py --device cuda --flex \
        2>&1 | tee "$RUN_GROUP/layer_adapter_check.log"
fi
# The parallel preflight launcher verifies compatibility before Target loading.
# Later groups reuse the same validation decisions, regardless of context mode.
checkpoints=()
for mode in "${modes[@]}"; do
    export OUTPUT_DIR="$RUN_GROUP/$mode" CONFIG="$RUN_GROUP/$mode/config.yaml"
    echo "Training $mode on $CUDA_VISIBLE_DEVICES: $EPOCHS epochs; micro=$MICRO_BATCH_SIZE; accumulation=$ACCUMULATION_STEPS"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    [[ -f "$OUTPUT_DIR/last/config.json" && -d "$OUTPUT_DIR/epoch_$EPOCHS" ]] || { echo "Missing final/epoch checkpoint for $mode" >&2; exit 1; }
    checkpoints+=(--checkpoint "$mode" "$OUTPUT_DIR/last")
done
export EVAL_GPUS="${CUDA_VISIBLE_DEVICES//,/ }"
export EVAL_SAMPLES=32 VIDEO_SAMPLES=32 EVAL_BATCH_SIZE=1 EVAL_MAX_NEW_TOKENS=256 SAMPLE_SEED=42
export EVAL_PROMPT_FIELD=reference VIDEO_FRAMES=16 VIDEO_MAX_PIXELS_PER_FRAME=200704
export RESULT_DIR="$RUN_GROUP/eval_all_b1_n32"
bash tests/evaluate_dflash_all.sh "${checkpoints[@]}" 2>&1 | tee "$RUN_GROUP/evaluate.log"
echo "Training and evaluation complete: $RUN_GROUP"
echo "Comparison: $RESULT_DIR/comparison.md"
