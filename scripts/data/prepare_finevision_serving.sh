#!/usr/bin/env bash
# CPU adapter + concurrent client; GPU model weights live in separate servers.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(dirname -- "$REPO_ROOT")"
CLIENT_PYTHON="${CLIENT_PYTHON:-$REPO_ROOT/.venv-vlm/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
SOURCE_DATA="${SOURCE_DATA:-$HOME/LMM/liyt/datasets/FineVision/FineVision_subset/finevision_sharegpt.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
PREPARE_OUTPUT="${PREPARE_OUTPUT:-$WORKSPACE/data/finevision_qwen35_sglang_${MAX_SAMPLES}}"
MANIFEST_DIR="${MANIFEST_DIR:-$PREPARE_OUTPUT/manifest}"
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

# Prefer positional addresses (safe with shell quoting); SERVER_ADDRESSES is
# also supported as a whitespace-separated list for ordinary local URLs.
if (( $# )); then
    servers=("$@")
else
    read -r -a servers <<< "${SERVER_ADDRESSES:-http://127.0.0.1:30000}"
fi
if (( ${#servers[@]} == 0 )); then
    echo 'At least one server address is required.' >&2
    exit 1
fi

"$CLIENT_PYTHON" scripts/data/prepare_finevision_manifest.py \
    --input "$SOURCE_DATA" \
    --output-dir "$MANIFEST_DIR" \
    --max-samples "$MAX_SAMPLES"

exec "$CLIENT_PYTHON" scripts/data/regenerate_vlm_serving.py \
    --manifest "$MANIFEST_DIR/prompts.jsonl" \
    --output-dir "$PREPARE_OUTPUT" \
    --target-model "$TARGET_MODEL" \
    --model "${SERVED_MODEL_NAME:-qwen35-vlm}" \
    --server-address "${servers[@]}" \
    --concurrency "${CONCURRENCY:-8}" \
    --max-tokens "${MAX_NEW_TOKENS:-256}" \
    --max-length "${MAX_LENGTH:-4096}" \
    --timeout "${REQUEST_TIMEOUT:-600}" \
    --retries "${REQUEST_RETRIES:-2}" \
    --image-max-tokens "${IMAGE_MAX_TOKENS:-0}" \
    --min-answer-tokens "${MIN_ANSWER_TOKENS:-2}"
