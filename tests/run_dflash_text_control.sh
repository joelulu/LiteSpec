#!/usr/bin/env bash
# PerfectBlend first turn -> target rollout -> same VLM trainer -> paired evaluation.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
cd "$REPO_ROOT"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export CLIENT_PYTHON="$PYTHON_BIN"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export SOURCE_DATA="${SOURCE_DATA:-$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/train_datasets/perfectblend_train.jsonl}"
export SGLANG_PYTHON="${SGLANG_PYTHON:-$HOME/.conda/envs/dflash/bin/python}"
export DATA_GPUS="${DATA_GPUS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DATA_GPUS}"
IFS=, read -r -a training_gpus <<< "$CUDA_VISIBLE_DEVICES"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#training_gpus[@]}}"
export MAX_SAMPLES="${TEXT_SAMPLES:-20000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export MIN_ANSWER_TOKENS="${MIN_ANSWER_TOKENS:-1}"
export CONCURRENCY="${CONCURRENCY:-8}"
export IMAGE_MAX_TOKENS=0
export PREPARE_OUTPUT="${PREPARE_OUTPUT:-$DLITE_WORKSPACE/data/perfectblend_qwen35_20k_text}"
export PROMPT_MANIFEST="$PREPARE_OUTPUT/selection/prompts.jsonl"
MODE="${MODE:-all}"
case "$MODE" in all|prepare|train|eval) ;; *) echo 'MODE=all|prepare|train|eval' >&2; exit 2 ;; esac
if [[ "$MODE" == all || "$MODE" == prepare ]]; then
    mkdir -p "$PREPARE_OUTPUT"
    "$PYTHON_BIN" tests/prepare_text_control.py --source "$SOURCE_DATA" \
        --output-dir "$PREPARE_OUTPUT/selection" --samples "$MAX_SAMPLES" --seed "${SAMPLE_SEED:-42}" \
        2>&1 | tee "$PREPARE_OUTPUT/prepare.log"
    "$CLIENT_PYTHON" -m litespec.vlm.data_pipeline 2>&1 | tee "$PREPARE_OUTPUT/rollout.log"
    [[ "$MODE" != prepare ]] || exit 0
fi
export TRAIN_DATA="$PREPARE_OUTPUT/train.jsonl"
if [[ "$MODE" == all || "$MODE" == train ]]; then
    [[ -s "$TRAIN_DATA" ]] || { echo "Missing training data: $TRAIN_DATA" >&2; exit 1; }
    export OUTPUT_DIR="${OUTPUT_DIR:-$DLITE_WORKSPACE/runs/dflash_text_20k_3ep_$(date +%Y%m%d_%H%M%S)_$$}"
    [[ ! -e "$OUTPUT_DIR" ]] || { echo "Choose a fresh OUTPUT_DIR: $OUTPUT_DIR" >&2; exit 1; }
    mkdir -p "$OUTPUT_DIR"
    export CONFIG="$OUTPUT_DIR/config.yaml"
    "$PYTHON_BIN" - "$CONFIG" <<'PY'
import os, sys, yaml
from pathlib import Path
config = yaml.safe_load(Path('scripts/config/vlm/sft_dflash.yaml').read_text())
config.update(num_epochs=int(os.getenv('EPOCHS', '3')),
              batch_size=int(os.getenv('MICRO_BATCH_SIZE', '2')),
              accumulation_steps=int(os.getenv('ACCUMULATION_STEPS', '4')),
              block_size=int(os.getenv('BLOCK_SIZE', '8')),
              max_length=int(os.environ['MAX_LENGTH']))
Path(sys.argv[1]).write_text(yaml.safe_dump(config, sort_keys=False))
PY
    export ALGORITHM=dflash REPORT_TO="${REPORT_TO:-tensorboard}" SAVE_TRAINING_STATE=0
    # The full random subset is selected above; avoid stale debug caps or image eligibility.
    unset MAX_STEPS MAX_SAMPLES REUSE_VALIDATION_FROM
    echo "Text training output: $OUTPUT_DIR"
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    export DFLASH_TEXT="$OUTPUT_DIR/last"
fi
if [[ "$MODE" == all || "$MODE" == eval ]]; then
    export DFLASH_TEXT="${DFLASH_TEXT:?Set DFLASH_TEXT for MODE=eval}"
    export DFLASH_VLM="${DFLASH_VLM:-$DLITE_WORKSPACE/runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dflash/last}"
    export MODELS="${MODELS:-ar dlite_original dflash_text dflash_vlm}"
    export TESTS="${TESTS:-text-only charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k}"
    export MODES=vision
    export EVAL_GPU="${EVAL_GPU:-${training_gpus[0]}}"
    export BATCH_SIZES="${BATCH_SIZES:-1}" EVAL_SAMPLES="${EVAL_SAMPLES:-16}"
    export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
    export RESULT_DIR="${RESULT_DIR:-${OUTPUT_DIR:-$(dirname -- "$DFLASH_TEXT")}/evaluation_$(date +%Y%m%d_%H%M%S)_$$}"
    bash tests/compare_static_batch_text_vlm.sh
    echo "Tables: $RESULT_DIR/speedup_tables.md"
fi
