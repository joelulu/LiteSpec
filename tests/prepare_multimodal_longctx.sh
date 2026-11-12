#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" tests/prepare_multimodal_longctx.py \
 --output-dir "${LONGCTX_SUITE_DIR:-$DLITE_WORKSPACE/data/multimodal_longctx100}" \
 --cache-dir "${LONGCTX_HF_CACHE:-$DLITE_WORKSPACE/data/hf_longctx_cache}" \
 --samples "${PREPARE_SAMPLES:-100}" --seed "${SAMPLE_SEED:-42}" "$@"
