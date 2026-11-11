#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" tests/prepare_longctx_data.py \
    --output-dir "${LONGCTX_DATA_DIR:-$DLITE_WORKSPACE/data/longctx_mathvista100}" \
    --cache-dir "${LONGCTX_HF_CACHE:-$DLITE_WORKSPACE/data/hf_longctx_cache}" --samples 100 "$@"
