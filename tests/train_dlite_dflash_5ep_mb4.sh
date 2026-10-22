#!/usr/bin/env bash
# Run here or copy this file into dlite-vlm/.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$HERE/LiteSpec" ]]; then
    DEFAULT_WORKSPACE="$HERE"
else
    DEFAULT_WORKSPACE="$(cd -- "$HERE/../.." && pwd)"
fi
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$DEFAULT_WORKSPACE}"
cd "$DLITE_WORKSPACE/LiteSpec"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export REPORT_TO=tensorboard
export SAVE_TRAINING_STATE=0
unset MAX_STEPS MAX_SAMPLES CONFIG OUTPUT_DIR ALGORITHM
[[ -s "$TRAIN_DATA" ]] || { echo "Missing/empty training data: $TRAIN_DATA" >&2; exit 1; }
"$PYTHON_BIN" -c 'from torch.utils.tensorboard import SummaryWriter; import yaml'
RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/vlm_dlite_dflash_5ep_mb4_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$RUN_GROUP" ]] || { echo "Choose a fresh RUN_GROUP: $RUN_GROUP" >&2; exit 1; }
mkdir -p "$RUN_GROUP"
echo "All training outputs: $RUN_GROUP"
for algorithm in dlite dflash; do
    export ALGORITHM="$algorithm"
    export OUTPUT_DIR="$RUN_GROUP/$algorithm"
    mkdir -p "$OUTPUT_DIR"
    if [[ "$algorithm" == dlite ]]; then
        base="scripts/config/vlm/sft_student_aligned.yaml"
    else
        base="scripts/config/vlm/sft_dflash.yaml"
    fi
    export CONFIG="$OUTPUT_DIR/config_5ep_mb4.yaml"
    "$PYTHON_BIN" - "$base" "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
config.update(num_epochs=5, batch_size=4, accumulation_steps=4)
Path(sys.argv[2]).write_text(yaml.safe_dump(config, sort_keys=False))
PY
    echo "Starting $algorithm: epochs=5, GPUs=8, micro batch=4, global batch=128"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    echo "Completed $algorithm: $OUTPUT_DIR/last"
done
echo "Completed both training runs: $RUN_GROUP"
