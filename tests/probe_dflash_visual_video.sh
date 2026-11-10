#!/usr/bin/env bash
# Fixed-prefix probes only: TextVQA and two videos; no training or throughput run.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}"
export RESULT_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/dflash_visual_probe_video_50_$(date +%Y%m%d_%H%M%S)_$$}"
read -r -a tests <<< "${TESTS:-textvqa Video-MME VideoDetailCaption}"
"$PYTHON_BIN" tests/check_sw_token_source.py
"$PYTHON_BIN" tests/probe_dflash_visual.py \
    --target-model "${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}" \
    --checkpoint full_2w "${FULL_2W:-$DLITE_WORKSPACE/runs/dflash_context_20k_1ep_20261110_041837_1299222/full/last}" \
    --checkpoint dflash_100k "${DFLASH_100K:-$DLITE_WORKSPACE/runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dflash/last}" \
    --tests "${tests[@]}" \
    --samples "${EVAL_SAMPLES:-50}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}" \
    --probes-per-sample "${PROBES_PER_SAMPLE:-8}" --stride "${PROBE_STRIDE:-16}" \
    --seed "${SAMPLE_SEED:-42}" --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}" \
    --video-data-root "${VIDEO_DATA_ROOT:-$HOME/LMM/liyt/datasets/eval_bench}" \
    --frames "${VIDEO_FRAMES:-16}" --max-pixels-per-frame "${VIDEO_MAX_PIXELS_PER_FRAME:-200704}" \
    --output-dir "$RESULT_DIR"
echo "Results: $RESULT_DIR/diagnostics.md"
