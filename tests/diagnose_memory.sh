#!/usr/bin/env bash
# Reuse trained vision checkpoint; no training is launched.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv-vlm/bin/python}"
RUN_GROUP="${RUN_GROUP:-$WORKSPACE/runs/dlite_memory_3ep_20261024_144948_3458849}"
RESULT_DIR="${RESULT_DIR:-$RUN_GROUP/memory_diagnosis_$(date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$REPO"
read -r -a datasets <<< "${TESTS:-charxiv_val textvqa}"
options=()
[[ "${PROBE_ONLY:-0}" == 1 ]] && options+=(--probe-only)
[[ -n "${VISION_CHECKPOINT:-}" ]] && options+=(--checkpoint "$VISION_CHECKPOINT")
exec "$PYTHON_BIN" tests/diagnose_memory.py --run-group "$RUN_GROUP" --output-dir "$RESULT_DIR" \
    --datasets "${datasets[@]}" --samples "${EVAL_SAMPLES:-16}" \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}" --probe-stride "${PROBE_STRIDE:-8}" \
    --gradient-samples "${GRADIENT_SAMPLES:-1}" "${options[@]}" "$@"
