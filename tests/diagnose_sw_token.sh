#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}"
"$PYTHON_BIN" tests/check_sw_token_source.py
extra=()
if [[ -n "${REUSE_PREFIXES_FROM:-}" ]]; then extra+=(--reuse-prefixes-from "$REUSE_PREFIXES_FROM"); fi
"$PYTHON_BIN" tests/diagnose_sw_token.py \
  --workspace "$DLITE_WORKSPACE" \
  --checkpoint "${INIT_CHECKPOINT:-$DLITE_WORKSPACE/runs/vlm_three_3ep_20261021_045040_2983971/dlite/last}" \
  --target-model "${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}" \
  --output-dir "${RESULT_DIR:-$DLITE_WORKSPACE/runs/sw_textvqa_diagnosis_$(date +%Y%m%d_%H%M%S)_$$}" \
  --samples "${EVAL_SAMPLES:-16}" --seed "${SAMPLE_SEED:-42}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}" \
  --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}" \
  --stride "${PROBE_STRIDE:-8}" --probes-per-sample "${PROBES_PER_SAMPLE:-16}" \
  --draft-dtype "${DRAFT_DTYPE:-target}" "${extra[@]}" "$@"

