#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
: "${RUN_GROUP:?Set RUN_GROUP to the completed six-model training directory}"
export RESULT_DIR="${RESULT_DIR:-$RUN_GROUP/eval_$(date +%Y%m%d_%H%M%S)_$$}"
export TESTS="${TESTS:-text-only charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k Video-MME VideoDetailCaption}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-16}" EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
export EVAL_GPU="${EVAL_GPU:-7}" BATCH_SIZES=1 WINDOW_SIZES=1
unset SW_RUN_GROUP
read -r -a modes <<< "${EXPERIMENTS:-A B C D1 D2 E}"
args=(--trained-window)
for mode in "${modes[@]}"; do args+=(--checkpoint "$mode" "$RUN_GROUP/$mode/last"); done
bash tests/evaluate_sw_token.sh "${args[@]}" "$@"
cat "$RESULT_DIR/comparison.md"
