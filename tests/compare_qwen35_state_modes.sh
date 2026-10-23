#!/usr/bin/env bash
# Same checkpoints/prompts, three HF hybrid-state recovery strategies.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export EVAL_GPU="${EVAL_GPU:-6}"
export BATCH_SIZES="${BATCH_SIZES:-1}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-16}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
export SAMPLE_SEED="${SAMPLE_SEED:-42}"
export MODELS="${MODELS:-ar dflash_vlm}"
export DFLASH_VLM="${DFLASH_VLM:-$DLITE_WORKSPACE/runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dflash/last}"
export TESTS="${TESTS:-text-only charxiv_val}"
export MODES="${MODES:-vision}"
export PROFILE_DECODE=1
export PROFILE_SAMPLES="${PROFILE_SAMPLES:-16}"
export PROFILE_KERNELS="${PROFILE_KERNELS:-1}"
GROUP_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/qwen35_state_modes_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$GROUP_DIR" ]] || { echo "Choose a fresh RESULT_DIR: $GROUP_DIR" >&2; exit 1; }
mkdir -p "$GROUP_DIR"
GROUP_DIR="$(cd -- "$GROUP_DIR" && pwd)"
cd "$REPO_ROOT"
# Catch Triton/device numerical problems before loading any large checkpoint.
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" tests/check_gdn_state_cuda.py \
    --width "${STATE_CHECK_WIDTH:-16}" | tee "$GROUP_DIR/cuda_state_check.json"
for mode in full_target_replay per_token compact_replay; do
    echo "=== GDN recovery: $mode ==="
    STATE_MODE="$mode" RESULT_DIR="$GROUP_DIR/$mode" \
        bash tests/compare_static_batch_text_vlm.sh 2>&1 | tee "$GROUP_DIR/$mode.log"
done
"$PYTHON_BIN" tests/report_qwen35_state_modes.py --root "$GROUP_DIR"
echo "Comparison: $GROUP_DIR/state_modes.md"
echo "Full metrics: $GROUP_DIR/state_modes.csv"
