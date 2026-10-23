#!/usr/bin/env bash
# Normal speed benchmark + separate timeline and actual-kernel diagnostic passes.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
export EVAL_GPU="${EVAL_GPU:-6}"
export BATCH_SIZES="${BATCH_SIZES:-1}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-16}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
export MODELS="${MODELS:-ar dflash_vlm}"
export DFLASH_VLM="${DFLASH_VLM:-$DLITE_WORKSPACE/runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dflash/last}"
export TESTS="${TESTS:-text-only charxiv_val}"
export MODES="${MODES:-vision}"
export PROFILE_DECODE=1
export PROFILE_SAMPLES="${PROFILE_SAMPLES:-4}"
export PROFILE_KERNELS="${PROFILE_KERNELS:-1}"
export RESULT_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/vlm_decode_profile_$(date +%Y%m%d_%H%M%S)_$$}"
cd "$REPO_ROOT"
bash tests/compare_static_batch_text_vlm.sh
echo "Speed: $RESULT_DIR/speedup_tables.md"
echo "Cost:  $RESULT_DIR/inference_profile.md"
