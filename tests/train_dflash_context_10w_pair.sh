#!/usr/bin/env bash
# Recorded 2026-11-10 configuration: 10w candidates, 2 epochs, mb4/acc16.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export TRAIN_DATA="$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl"
export MAX_SAMPLES=0 EPOCHS=2 MICRO_BATCH_SIZE=4 ACCUMULATION_STEPS=16 IMAGE_MAX_TOKENS=1024
export REUSE_VALIDATION_FROM="${VALIDATION_FROM:-$DLITE_WORKSPACE/runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb}"
export RUN_PREFIX=dflash_full_sw1_10w
unset INIT_CHECKPOINT
bash "$REPO/tests/train_dflash_context_pair.sh"
