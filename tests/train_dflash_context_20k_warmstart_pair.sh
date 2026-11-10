#!/usr/bin/env bash
# Both modes adapt the SAME old 10w Full checkpoint on the SAME 20k dataset.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export TRAIN_DATA="$DLITE_WORKSPACE/data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl"
export MAX_SAMPLES=20000 EPOCHS=2 MICRO_BATCH_SIZE=4 ACCUMULATION_STEPS=16 IMAGE_MAX_TOKENS=1024
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-$DLITE_WORKSPACE/runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dflash/last}"
export REUSE_VALIDATION_FROM="${VALIDATION_FROM:-$DLITE_WORKSPACE/runs/dflash_context_20k_1ep_20261110_041837_1299222/full}"
export RUN_PREFIX=dflash_full_sw1_20k_from10w
bash "$REPO/tests/train_dflash_context_pair.sh"
