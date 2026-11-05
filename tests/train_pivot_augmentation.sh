#!/usr/bin/env bash
# Six from-scratch experiments, eight GPUs per run; final weights only.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8 SAVE_TRAINING_STATE=0 ALGORITHM=dlite
export REPORT_TO="${REPORT_TO:-tensorboard}"
export EPOCHS="${EPOCHS:-2}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}" ROLLOUT_ANCHORS="${ROLLOUT_ANCHORS:-4}"
export IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
export MAX_SAMPLES="${MAX_SAMPLES:-20000}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
unset INIT_CHECKPOINT RESUME CONFIG OUTPUT_DIR SW_RUN_GROUP MAX_STEPS
if [[ -n "${DEBUG_MAX_STEPS:-}" ]]; then export MAX_STEPS="$DEBUG_MAX_STEPS"; fi
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/pivot_aug_20k_2ep_$(date +%Y%m%d_%H%M%S)_$$}"
[[ -s "$TRAIN_DATA" ]] || { echo "Missing data: $TRAIN_DATA" >&2; exit 1; }
[[ ! -e "$RUN_GROUP" ]] || { echo "Choose fresh RUN_GROUP: $RUN_GROUP" >&2; exit 1; }
"$PYTHON_BIN" tests/check_sw_token_source.py
"$PYTHON_BIN" - <<'PY'
import os, torch
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 8, 'Select exactly eight GPUs'
assert torch.cuda.device_count() == 8, 'Eight visible GPUs required'
if os.environ['REPORT_TO'] == 'tensorboard':
    from torch.utils.tensorboard import SummaryWriter
PY
read -r -a modes <<< "${EXPERIMENTS:-A B C D1 D2 E}"
"$PYTHON_BIN" - "${modes[@]}" <<'PY'
import sys
from litespec.vlm.pivot_augmentation import MODES
assert sys.argv[1:] and len(set(sys.argv[1:])) == len(sys.argv[1:])
assert all(x in MODES for x in sys.argv[1:]), MODES
PY
mkdir -p "$RUN_GROUP"
reuse="${REUSE_VALIDATION_FROM:-}"
echo "Output: $RUN_GROUP; all runs FROM SCRATCH, eight GPUs, final export only"
echo "Independent two-round anchors/sample=$ROLLOUT_ANCHORS (not upstream 512 teacher-forced anchors)"
for mode in "${modes[@]}"; do
    export OUTPUT_DIR="$RUN_GROUP/$mode" CONFIG="$RUN_GROUP/$mode/config.yaml"
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" - "$mode" <<'PY'
import os, sys, yaml
from pathlib import Path
c=yaml.safe_load(Path('scripts/config/vlm/sft_student_aligned.yaml').read_text())
c.update(pivot_two_rounds=True, final_only=True, context_window_size=1,
    pivot_augmentation=dict(mode=sys.argv[1], rank=128, history_positions=4),
    num_epochs=int(os.environ['EPOCHS']), batch_size=int(os.environ['MICRO_BATCH_SIZE']),
    accumulation_steps=int(os.environ['ACCUMULATION_STEPS']),
    rollout_anchors=int(os.environ['ROLLOUT_ANCHORS']), num_anchors=int(os.environ['ROLLOUT_ANCHORS']), block_size=8,
    num_draft_layers=5, chs_num_layers=12, sequential_rank=320,
    target_layer_ids=[0,1,4,7,11,14,17,20,24,27,30,31],
    image_max_tokens=int(os.environ['IMAGE_MAX_TOKENS']), max_length=4096,
    report_to=os.environ['REPORT_TO'], save_interval=10**12, log_interval=1,
    final_save=True, seed=42)
Path(os.environ['CONFIG']).write_text(yaml.safe_dump(c,sort_keys=False))
PY
    if [[ -n "$reuse" ]]; then export REUSE_VALIDATION_FROM="$reuse"; else unset REUSE_VALIDATION_FROM; fi
    echo "Starting $mode: $EPOCHS epochs, 8 GPUs, micro=$MICRO_BATCH_SIZE, accumulation=$ACCUMULATION_STEPS"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    [[ -f "$OUTPUT_DIR/last/config.json" ]] || { echo "Missing final model" >&2; exit 1; }
    reuse="$OUTPUT_DIR"
done
echo "All final models: $RUN_GROUP/{A,B,C,D1,D2,E}/last"
