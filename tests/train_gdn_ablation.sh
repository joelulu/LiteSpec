#!/usr/bin/env bash
# Four matched from-scratch arms on physical GPUs 4-7, then automatic evaluation.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export REPORT_TO=tensorboard
export SAVE_TRAINING_STATE="${SAVE_TRAINING_STATE:-0}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
export EPOCHS="${EPOCHS:-3}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
export GDN_LAYERS="${GDN_LAYERS:-14,30}"
export GDN_RANK="${GDN_RANK:-8}"
export GDN_BOTTLENECK="${GDN_BOTTLENECK:-128}"
export GDN_CHUNK_ANCHORS="${GDN_CHUNK_ANCHORS:-16}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export LITESPEC_EAGER_BLOCK_MASK=1
# Debug caps must be explicitly opted into; do not inherit another experiment.
unset MAX_STEPS MAX_SAMPLES CONFIG OUTPUT_DIR ALGORITHM REUSE_VALIDATION_FROM
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
    export MAX_STEPS=1 MAX_SAMPLES=32
    export EVAL_SAMPLES="${EVAL_SAMPLES:-2}" EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-32}"
fi
if [[ -n "${PREPARED_VALIDATION_FROM:-}" && "${SMOKE_TEST:-0}" != 1 ]]; then
    export REUSE_VALIDATION_FROM="$PREPARED_VALIDATION_FROM"
fi
cd "$REPO_ROOT"
[[ -s "$TRAIN_DATA" ]] || { echo "Missing/empty training data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "Missing target: $TARGET_MODEL" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
"$PYTHON_BIN" -c 'import torch, transformers, yaml, triton, causal_conv1d; from torch.utils.tensorboard import SummaryWriter; assert torch.cuda.device_count() == 4, "Need physical GPUs 4,5,6,7"'
RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/vlm_gdn_${EPOCHS}ep_mb${MICRO_BATCH_SIZE}_b${BLOCK_SIZE}_$(date +%Y%m%d_%H%M%S)_$$}"
export RUN_GROUP
[[ ! -e "$RUN_GROUP" ]] || { echo "Choose a fresh RUN_GROUP: $RUN_GROUP" >&2; exit 1; }
mkdir -p "$RUN_GROUP"
echo "Run group: $RUN_GROUP"
echo "GPUs=$CUDA_VISIBLE_DEVICES, per-GPU=$MICRO_BATCH_SIZE, accumulation=$ACCUMULATION_STEPS, global batch=$((4 * MICRO_BATCH_SIZE * ACCUMULATION_STEPS))"
# Validate the sparse CUDA capture and full native image training/reload path on
# each participating GPU BEFORE consuming the training dataset.
for gpu in 4 5 6 7; do
    echo "Starting CUDA preflight on GPU $gpu: $RUN_GROUP/cuda_check_$gpu.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tests/check_gdn_reader_cuda.py \
        > "$RUN_GROUP/cuda_check_$gpu.log" 2>&1 || { cat "$RUN_GROUP/cuda_check_$gpu.log"; exit 1; }
    echo "Completed CUDA preflight on GPU $gpu"
done
for arm in dflash dlite dflash_gdn dlite_gdn; do
    export ALGORITHM="${arm%_gdn}"
    export OUTPUT_DIR="$RUN_GROUP/$arm"
    mkdir -p "$OUTPUT_DIR"
    export CONFIG="$OUTPUT_DIR/config.yaml"
    "$PYTHON_BIN" - "$arm" "$CONFIG" <<'PY'
from pathlib import Path
import os
import sys
import yaml
arm = sys.argv[1]
family = arm.removesuffix('_gdn')
base = 'sft_student_aligned.yaml' if family == 'dlite' else 'sft_dflash.yaml'
cfg = yaml.safe_load((Path('scripts/config/vlm')/base).read_text())
cfg.update(algorithm=family, gdn_experiment=True, gdn_enabled=arm.endswith('_gdn'),
           num_epochs=int(os.environ['EPOCHS']), batch_size=int(os.environ['MICRO_BATCH_SIZE']),
           accumulation_steps=int(os.environ['ACCUMULATION_STEPS']), block_size=int(os.environ['BLOCK_SIZE']),
           image_max_tokens=int(os.environ['IMAGE_MAX_TOKENS']), max_length=int(os.environ['MAX_LENGTH']), seed=42,
           num_draft_layers=5, num_anchors=512,
           gdn_layer_ids=[int(i) for i in os.environ['GDN_LAYERS'].split(',')],
           gdn_rank=int(os.environ['GDN_RANK']), gdn_bottleneck=int(os.environ['GDN_BOTTLENECK']),
           gdn_chunk_anchors=int(os.environ['GDN_CHUNK_ANCHORS']), gdn_injection_layer=4)
Path(sys.argv[2]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
    echo "Starting $arm (from scratch): $OUTPUT_DIR"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    [[ -s "$OUTPUT_DIR/last/config.json" ]] || { echo "No completed checkpoint: $arm" >&2; exit 1; }
    # Only a successful torchrun exit creates this marker. Failed/partial runs
    # cannot enter evaluation as completed training arms.
    "$PYTHON_BIN" - "$OUTPUT_DIR" "${SMOKE_TEST:-0}" <<'PY'
from pathlib import Path
import json
import sys
p = Path(sys.argv[1])
checkpoint = json.loads((p/'last/checkpoint.json').read_text())
(p/'training_complete.json').write_text(json.dumps(dict(checkpoint=str((p/'last').resolve()),
    cursor=checkpoint['cursor'], smoke_test=sys.argv[2] == '1'), indent=2)+'\n')
PY
    export REUSE_VALIDATION_FROM="$OUTPUT_DIR"
    echo "Completed $arm: $OUTPUT_DIR/last"
done
echo "All four training arms succeeded. Starting automatic evaluation on GPUs 4-7."
bash tests/evaluate_gdn_ablation.sh "$RUN_GROUP" 2>&1 | tee "$RUN_GROUP/eval.log"
echo "Training and evaluation finished: $RUN_GROUP/evaluation/matrix.md"
