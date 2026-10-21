#!/usr/bin/env bash
# Text, multimodal train and multimodal validation; no correctness gate.
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
TRAIN_EVAL_DATA="${TRAIN_EVAL_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/train.jsonl}"
VAL_EVAL_DATA="${VAL_EVAL_DATA:-${EVAL_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/val.jsonl}}"
GROUP_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/text_vlm_compare_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$GROUP_DIR" ]] || { echo "Choose a fresh RESULT_DIR" >&2; exit 1; }
mkdir -p "$GROUP_DIR"
GROUP_DIR="$(cd -- "$GROUP_DIR" && pwd)"
cd "$REPO_ROOT"
# Catch accidentally testing image prompts as text or vice versa before loading weights.
"$PYTHON_BIN" - "$TEXT_DATA" "$TRAIN_EVAL_DATA" "$VAL_EVAL_DATA" "$EVAL_SAMPLES" <<'PY'
import json, sys
from itertools import islice
from pathlib import Path
assert Path(sys.argv[2]).resolve() != Path(sys.argv[3]).resolve(), 'train and validation manifests must differ'
for path, image_expected in zip(sys.argv[1:4], (False, True, True)):
    with open(path) as stream:
        rows = list(islice((json.loads(line) for line in stream if line.strip()), int(sys.argv[4])))
    assert len(rows) == int(sys.argv[4]), f'not enough samples: {path}'
    assert all(bool(row.get('image')) == image_expected for row in rows), f'wrong input modality: {path}'
PY
for stage in text multimodal_train multimodal_val; do
    case "$stage" in
        text) data="$TEXT_DATA"; split=smoke ;;
        multimodal_train) data="$TRAIN_EVAL_DATA"; split=train ;;
        multimodal_val) data="$VAL_EVAL_DATA"; split=validation ;;
    esac
    echo "=== $stage: split=$split, data=$data, C=$CONCURRENCIES, $EVAL_SAMPLES requests per cell ==="
    EVAL_DATA="$data" EVAL_SPLIT="$split" RESULT_DIR="$GROUP_DIR/$stage" \
        bash tests/compare_vlm_16.sh 2>&1 | tee "$GROUP_DIR/$stage.log"
done
"$PYTHON_BIN" tests/report_text_vlm_speedup.py --root "$GROUP_DIR"
echo "Tables: $GROUP_DIR/speedup_tables.md"
echo "Full metrics: $GROUP_DIR/comparison.csv"
