#!/usr/bin/env bash
# Run in place, or copy into the parent dlite-vlm workspace.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$HERE/LiteSpec" ]]; then
    DEFAULT_WORKSPACE="$HERE"
else
    DEFAULT_WORKSPACE="$(cd -- "$HERE/../.." && pwd)"
fi
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$DEFAULT_WORKSPACE}"
REPO_ROOT="$DLITE_WORKSPACE/LiteSpec"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/train.jsonl}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export REPORT_TO="${REPORT_TO:-tensorboard}"
export SAVE_TRAINING_STATE=0
unset MAX_STEPS MAX_SAMPLES CONFIG OUTPUT_DIR ALGORITHM
cd "$REPO_ROOT"
[[ -s "$TRAIN_DATA" ]] || { echo "Missing training data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "Missing target model: $TARGET_MODEL" >&2; exit 1; }
"$PYTHON_BIN" - "$REPORT_TO" <<'PY'
import sys
import yaml
if sys.argv[1] == 'tensorboard':
    from torch.utils.tensorboard import SummaryWriter
PY
RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/vlm_three_3ep_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$RUN_GROUP" ]] || { echo "Choose a new RUN_GROUP: $RUN_GROUP" >&2; exit 1; }
mkdir -p "$RUN_GROUP"
RUN_GROUP="$(cd -- "$RUN_GROUP" && pwd)"
echo "All training outputs: $RUN_GROUP"
for algorithm in dlite dflash dspark; do
    export ALGORITHM="$algorithm"
    export OUTPUT_DIR="$RUN_GROUP/$algorithm"
    mkdir -p "$OUTPUT_DIR"
    if [[ "$algorithm" == dlite ]]; then
        base="$REPO_ROOT/scripts/config/vlm/sft_student_aligned.yaml"
    else
        base="$REPO_ROOT/scripts/config/vlm/sft_${algorithm}.yaml"
    fi
    export CONFIG="$OUTPUT_DIR/config_3ep.yaml"
    "$PYTHON_BIN" - "$base" "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
config['num_epochs'] = 3
Path(sys.argv[2]).write_text(yaml.safe_dump(config, sort_keys=False))
PY
    echo "Starting $algorithm: epochs=3, GPUs=8; output=$OUTPUT_DIR"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    echo "Completed $algorithm: $OUTPUT_DIR/last"
done
echo "Completed all three training runs: $RUN_GROUP"
