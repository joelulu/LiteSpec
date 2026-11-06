#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}"
PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
: "${RUN_GROUP:?Set RUN_GROUP to the completed D0/D1/D2 training directory}"
args=()
read -r -a modes <<< "${EXPERIMENTS:-D0 D1 D2}"
for mode in "${modes[@]}"; do args+=(--checkpoint "$mode" "$RUN_GROUP/$mode/last"); done
read -r -a datasets <<< "${TESTS:-charxiv_val textvqa text-only}"
"$PYTHON_BIN" tests/check_sw_token_source.py
"$PYTHON_BIN" tests/diagnose_pivot_feedback.py \
 --target-model "${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}" \
 --output-dir "${RESULT_DIR:-$RUN_GROUP/feedback_probe_$(date +%Y%m%d_%H%M%S)_$$}" \
 --tests "${datasets[@]}" --samples "${EVAL_SAMPLES:-16}" \
 --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-128}" --probes-per-sample "${PROBES_PER_SAMPLE:-8}" \
 --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}" --seed "${SAMPLE_SEED:-42}" "${args[@]}" "$@"
