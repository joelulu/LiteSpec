#!/usr/bin/env bash
# Two independent comparison groups, eight GPUs per run; final weights only.
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
export EPOCHS="${EPOCHS:-2}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}" ROLLOUT_ANCHORS="${ROLLOUT_ANCHORS:-4}"
export NUM_ANCHORS="${NUM_ANCHORS:-512}" SFT_CHUNK_BLOCKS="${SFT_CHUNK_BLOCKS:-16}"
export IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
export MAX_SAMPLES="${MAX_SAMPLES:-20000}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
unset INIT_CHECKPOINT RESUME CONFIG OUTPUT_DIR SW_RUN_GROUP MAX_STEPS
if [[ -n "${DEBUG_MAX_STEPS:-}" ]]; then export MAX_STEPS="$DEBUG_MAX_STEPS"; fi
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/pivot_split_20k_2ep_$(date +%Y%m%d_%H%M%S)_$$}"
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
case "${TRAIN_SET:-all}" in
    all) defaults="A B C E D0 D1 D2" ;;
    sft) defaults="A B C E" ;;
    rollout) defaults="D0 D1 D2" ;;
    *) echo "TRAIN_SET must be all, sft or rollout" >&2; exit 2 ;;
esac
read -r -a modes <<< "${EXPERIMENTS:-$defaults}"
"$PYTHON_BIN" - "${modes[@]}" <<'PY'
import sys
from litespec.vlm.pivot_augmentation import MODES
assert sys.argv[1:] and len(set(sys.argv[1:])) == len(sys.argv[1:])
assert all(x in MODES for x in sys.argv[1:]), MODES
PY
mkdir -p "$RUN_GROUP"
declare -A reuse
reuse[sft]="${SFT_REUSE_VALIDATION_FROM:-}"
reuse[rollout]="${ROLLOUT_REUSE_VALIDATION_FROM:-}"
if [[ -n "${REUSE_VALIDATION_FROM:-}" ]]; then
    reuse_kind="$("$PYTHON_BIN" - "$REUSE_VALIDATION_FROM" <<'PY'
import json,sys
from pathlib import Path
c=json.loads((Path(sys.argv[1])/'effective_config.json').read_text())
print('rollout' if c.get('pivot_two_rounds') else 'sft')
PY
)"
    reuse[$reuse_kind]="$REUSE_VALIDATION_FROM"
fi
echo "Output: $RUN_GROUP; FROM SCRATCH, eight GPUs, final export only"
echo "A/B/C/E: one shared Target pass, up to $NUM_ANCHORS anchors/sample"
echo "D0/D1/D2: cached two-round rollout, $ROLLOUT_ANCHORS independent anchors/sample"
for mode in "${modes[@]}"; do
    export OUTPUT_DIR="$RUN_GROUP/$mode" CONFIG="$RUN_GROUP/$mode/config.yaml"
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" tests/pivot_screen_config.py --mode "$mode" --output "$CONFIG"
    case "$mode" in A|B|C|E) kind=sft ;; *) kind=rollout ;; esac
    if [[ -n "${reuse[$kind]}" ]]; then export REUSE_VALIDATION_FROM="${reuse[$kind]}"; else unset REUSE_VALIDATION_FROM; fi
    echo "Starting $mode: $EPOCHS epochs, 8 GPUs, micro=$MICRO_BATCH_SIZE, accumulation=$ACCUMULATION_STEPS"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    [[ -f "$OUTPUT_DIR/last/config.json" ]] || { echo "Missing final model" >&2; exit 1; }
    reuse[$kind]="$OUTPUT_DIR"
done
echo "Final models: $RUN_GROUP/{A,B,C,E,D0,D1,D2}/last"
