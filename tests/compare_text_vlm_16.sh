#!/usr/bin/env bash
# Independent text/image comparisons, no correctness gate between stages.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export ORIGINAL_DLITE="${ORIGINAL_DLITE:-$HOME/LMM/lmm_model/Qwen3.5-4B-DLite}"
export CONCURRENCIES="${CONCURRENCIES:-1 4 8}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-16}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
export REPORT_RAW_SPEEDUP=1
export VERIFICATION=block
# Always include the trained drafts plus the original DLite control.
export ONLY_ORIGINAL=0
TEXT_DATA="${TEXT_DATA:-$REPO_ROOT/tests/data/text_smoke_16.jsonl}"
IMAGE_DATA="${EVAL_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/val.jsonl}"
GROUP_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/text_vlm_compare_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$GROUP_DIR" ]] || { echo "Choose a fresh RESULT_DIR" >&2; exit 1; }
mkdir -p "$GROUP_DIR"
GROUP_DIR="$(cd -- "$GROUP_DIR" && pwd)"
cd "$REPO_ROOT"
# Catch accidentally testing image prompts as text or vice versa before loading weights.
"$PYTHON_BIN" - "$TEXT_DATA" "$IMAGE_DATA" "$EVAL_SAMPLES" <<'PY'
import json, sys
from itertools import islice
for path, image_expected in zip(sys.argv[1:3], (False, True)):
    with open(path) as stream:
        rows = list(islice((json.loads(line) for line in stream if line.strip()), int(sys.argv[3])))
    assert len(rows) == int(sys.argv[3]), f'not enough samples: {path}'
    assert all(bool(row.get('image')) == image_expected for row in rows), f'wrong input modality: {path}'
PY
for modality in text multimodal; do
    data="$TEXT_DATA"
    [[ "$modality" != multimodal ]] || data="$IMAGE_DATA"
    echo "=== $modality: C=$CONCURRENCIES, $EVAL_SAMPLES requests per cell ==="
    EVAL_DATA="$data" RESULT_DIR="$GROUP_DIR/$modality" \
        bash tests/compare_vlm_16.sh 2>&1 | tee "$GROUP_DIR/$modality.log"
done
"$PYTHON_BIN" tests/report_text_vlm_speedup.py --root "$GROUP_DIR"
echo "Tables: $GROUP_DIR/speedup_tables.md"
echo "Full metrics: $GROUP_DIR/comparison.csv"
