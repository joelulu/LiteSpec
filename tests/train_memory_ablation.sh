#!/usr/bin/env bash
# Three matched training runs, then four evaluation conditions.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv-vlm/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export SAVE_TRAINING_STATE="${SAVE_TRAINING_STATE:-0}"
export LITESPEC_EAGER_BLOCK_MASK="${LITESPEC_EAGER_BLOCK_MASK:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
export EPOCHS="${EPOCHS:-3}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
export MEMORY_MAX_TOKENS="${MEMORY_MAX_TOKENS:-1024}"
export MEMORY_RANK="${MEMORY_RANK:-256}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/dlite_memory_3ep_$(date +%Y%m%d_%H%M%S)_$$}"
cd "$REPO"
if [[ "${DEBUG_RUN:-0}" != 1 && ( -n "${MAX_STEPS:-}" || -n "${MAX_SAMPLES:-}" ) ]]; then
    echo "Unset MAX_STEPS/MAX_SAMPLES for full training, or set DEBUG_RUN=1 for an intentional smoke run." >&2
    exit 2
fi
[[ -s "$TRAIN_DATA" ]] || { echo "Missing data: $TRAIN_DATA" >&2; exit 1; }
[[ ! -e "$RUN_GROUP" ]] || { echo "Choose a fresh RUN_GROUP: $RUN_GROUP" >&2; exit 1; }
"$PYTHON_BIN" -c 'import yaml; from torch.utils.tensorboard import SummaryWriter'
mkdir -p "$RUN_GROUP"
echo "Outputs: $RUN_GROUP"
reuse="${REUSE_VALIDATION_FROM:-}"
for mode in none vision text; do
    export ALGORITHM=dlite
    export OUTPUT_DIR="$RUN_GROUP/$mode"
    export CONFIG="$OUTPUT_DIR/config.yaml"
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" - "$CONFIG" "$mode" <<'PY'
import os, sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path('scripts/config/vlm/sft_student_aligned.yaml').read_text())
for key, env in [('num_epochs','EPOCHS'), ('batch_size','MICRO_BATCH_SIZE'),
                 ('accumulation_steps','ACCUMULATION_STEPS'), ('block_size','BLOCK_SIZE'),
                 ('image_max_tokens','IMAGE_MAX_TOKENS'), ('memory_max_tokens','MEMORY_MAX_TOKENS'),
                 ('memory_rank','MEMORY_RANK'), ('seed','TRAIN_SEED')]:
    cfg[key] = int(os.environ[env])
cfg['memory_mode'] = sys.argv[2]
Path(sys.argv[1]).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"{sys.argv[2]}: epochs={cfg['num_epochs']}, micro={cfg['batch_size']}, "
      f"global batch={cfg['batch_size']*cfg['accumulation_steps']*int(os.environ['NPROC_PER_NODE'])}")
PY
    if [[ -n "$reuse" ]]; then export REUSE_VALIDATION_FROM="$reuse"; else unset REUSE_VALIDATION_FROM; fi
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    reuse="$OUTPUT_DIR"
done
if [[ "${RUN_EVAL:-1}" == 1 ]]; then
    "$PYTHON_BIN" tests/evaluate_memory_ablation.py --run-group "$RUN_GROUP"
fi
echo "Completed: $RUN_GROUP"
