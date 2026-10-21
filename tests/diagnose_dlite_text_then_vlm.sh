#!/usr/bin/env bash
# Text first: original native decoding and our reference. Gate image evaluation.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
ORIGINAL_DLITE="${ORIGINAL_DLITE:-$HOME/LMM/lmm_model/Qwen3.5-4B-DLite}"
TEXT_DATA="${TEXT_DATA:-$REPO_ROOT/tests/data/text_smoke_16.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/val.jsonl}"
RESULT_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/dlite_text_then_vlm_$(date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
[[ ! -e "$RESULT_DIR" ]] || { echo "Choose a fresh RESULT_DIR" >&2; exit 1; }
mkdir -p "$RESULT_DIR"
RESULT_DIR="$(cd -- "$RESULT_DIR" && pwd)"
cd "$REPO_ROOT"
run_case() {
    local stage="$1" execution="$2" data="$3"
    echo "Stage: $stage"
    if "$PYTHON_BIN" tests/benchmark_vlm_checkpoints.py \
        --target-model "$TARGET_MODEL" --checkpoint dlite_original "$ORIGINAL_DLITE" \
        --data "$data" --output-dir "$RESULT_DIR/$stage" \
        --max-samples "${EVAL_SAMPLES:-16}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-128}" \
        --concurrency 1 --warmup "${WARMUP:-1}" --verification block --execution "$execution" \
        2>&1 | tee "$RESULT_DIR/$stage.log"; then
        return 0
    fi
    # Mismatch results are retained for diagnosis. Setup/runtime failures abort.
    [[ -s "$RESULT_DIR/$stage/summary.json" ]] || return 1
    "$PYTHON_BIN" - "$RESULT_DIR/$stage/summary.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
assert any(row['model'] == 'dlite_original' for row in rows), 'checkpoint execution failed before producing a result'
PY
}
run_case text_native native_text "$TEXT_DATA"
run_case text_reference reference "$TEXT_DATA"
if "$PYTHON_BIN" tests/report_dlite_modalities.py --root "$RESULT_DIR" --check-text; then
    run_case multimodal_reference reference "$EVAL_DATA"
elif [[ "${RUN_MULTIMODAL_ALWAYS:-0}" == 1 ]]; then
    echo "Explicit override: collecting multimodal diagnostics despite failed text gate"
    run_case multimodal_reference reference "$EVAL_DATA"
else
    echo "Stopped before multimodal evaluation. Text diagnostics: $RESULT_DIR"
    exit 1
fi
"$PYTHON_BIN" tests/report_dlite_modalities.py --root "$RESULT_DIR"
echo "Results: $RESULT_DIR/analysis.csv"
