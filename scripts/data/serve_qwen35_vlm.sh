#!/usr/bin/env bash
# Run with a separate, Qwen3.5-capable SGLang Python environment.
set -euo pipefail

TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

if ! "$SGLANG_PYTHON" -c 'import importlib.util; assert importlib.util.find_spec("sglang")' >/dev/null 2>&1; then
    echo 'SGLang not found. Set SGLANG_PYTHON to a separate SGLang environment python.' >&2
    echo 'Do not install SGLang into LiteSpec/.venv-vlm.' >&2
    exit 1
fi

echo "SGLang Python: $SGLANG_PYTHON; GPU: $CUDA_VISIBLE_DEVICES; port: ${PORT:-30000}"
exec "$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --served-model-name "${SERVED_MODEL_NAME:-qwen35-vlm}" \
    --host "${HOST:-127.0.0.1}" \
    --port "${PORT:-30000}" \
    --tp-size "${TP_SIZE:-1}" \
    --dtype bfloat16 \
    --context-length "${CONTEXT_LENGTH:-4096}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}" \
    "$@"
