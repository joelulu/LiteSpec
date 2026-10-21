#!/usr/bin/env bash
# Selectable text, multimodal train and validation; no correctness gate.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
cd "$REPO_ROOT"
export ORIGINAL_DLITE="${ORIGINAL_DLITE:-$HOME/LMM/lmm_model/Qwen3.5-4B-DLite}"
export CONCURRENCIES="${CONCURRENCIES:-1 4 8}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-16}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
export REPORT_RAW_SPEEDUP=1
export VERIFICATION=block
export MODELS="${MODELS-ar dlite_original dflash_3ep dspark_3ep dlite_3ep dlite_6ep}"
TEXT_DATA="${TEXT_DATA:-$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main-0708-new/eval_datasets/gsm8k.jsonl}"
TRAIN_EVAL_DATA="${TRAIN_EVAL_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/train.jsonl}"
VAL_EVAL_DATA="${VAL_EVAL_DATA:-${EVAL_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/val.jsonl}}"
read -r -a selected_tests <<< "${TESTS-text-only visual-eval visual-training}"
((${#selected_tests[@]})) || { echo "TESTS must not be empty" >&2; exit 1; }
DATA_DIR="${DATA_DIR:-$HOME/LMM/liyt/datasets/jsonl_datas}"
export IMAGE_ROOT="${IMAGE_ROOT:-$DATA_DIR/imgs}"
export SAMPLE_SEED="${SAMPLE_SEED:-42}"
export PROMPT_FIELD="${PROMPT_FIELD:-reference}"
# visual-datasets expands only explicitly selected local evaluation datasets.
expanded_tests=()
for test_name in "${selected_tests[@]}"; do
    if [[ "$test_name" == visual-datasets ]]; then
        read -r -a datasets <<< "${DATASETS:-charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k}"
        ((${#datasets[@]})) || { echo "DATASETS must not be empty" >&2; exit 1; }
        expanded_tests+=("${datasets[@]}")
    else
        expanded_tests+=("$test_name")
    fi
done
selected_tests=("${expanded_tests[@]}")
stages=()
data_paths=()
splits=()
dataset_names=()
reference_inputs=()
input_modes=()
declare -A seen_tests=()
for test_name in "${selected_tests[@]}"; do
    [[ ! ${seen_tests[$test_name]+yes} ]] || { echo "Duplicate test: $test_name" >&2; exit 1; }
    seen_tests[$test_name]=1
    case "$test_name" in
        text-only)
            stages+=(text); data_paths+=("$TEXT_DATA"); splits+=("${TEXT_SPLIT:-unspecified}")
            dataset_names+=("${TEXT_DATASET_NAME:-gsm8k}"); reference_inputs+=(0); input_modes+=(text) ;;
        visual-eval)
            stages+=(multimodal_val); data_paths+=("$VAL_EVAL_DATA"); splits+=(validation)
            dataset_names+=(finevision); reference_inputs+=(0); input_modes+=(vision) ;;
        visual-training)
            stages+=(multimodal_train); data_paths+=("$TRAIN_EVAL_DATA"); splits+=(train)
            dataset_names+=(finevision); reference_inputs+=(0); input_modes+=(vision) ;;
        charxiv_val|flickr30k|textvqa|hr_bench_4k|hr_bench_8k)
            [[ "${BENCHMARK_MODE:-reference}" == static_batch ]] || { echo "Named local datasets require compare_static_batch_text_vlm.sh" >&2; exit 1; }
            data="$("$PYTHON_BIN" tests/reference_vlm_data.py --data-dir "$DATA_DIR" --resolve-dataset "$test_name")"
            read -r -a modes <<< "${MODES:-vision}"
            ((${#modes[@]})) || { echo "MODES must not be empty" >&2; exit 1; }
            declare -A seen_modes=()
            for mode in "${modes[@]}"; do
                case "$mode" in text|vision) ;; *) echo "Unknown mode: $mode" >&2; exit 1 ;; esac
                [[ ! ${seen_modes[$mode]+yes} ]] || { echo "Duplicate mode: $mode" >&2; exit 1; }
                seen_modes[$mode]=1
                stages+=("${test_name}_${mode}"); data_paths+=("$data"); splits+=(unspecified)
                dataset_names+=("$test_name"); reference_inputs+=(1); input_modes+=("$mode")
            done ;;
        *) echo "Unknown test: $test_name (choose text-only visual-eval visual-training visual-datasets charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k)" >&2; exit 1 ;;
    esac
done
if [[ ${seen_tests[visual-eval]+yes} && ${seen_tests[visual-training]+yes} ]]; then
    [[ "$TRAIN_EVAL_DATA" != "$VAL_EVAL_DATA" ]] || { echo "Training and validation manifests must differ" >&2; exit 1; }
fi
RUN_KIND=text_vlm_compare
AXIS="C=$CONCURRENCIES"
if [[ "${BENCHMARK_MODE:-reference}" == static_batch ]]; then
    RUN_KIND=static_batch_text_vlm
    AXIS="B=${BATCH_SIZES:-1 8}"
fi
GROUP_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/${RUN_KIND}_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$GROUP_DIR" ]] || { echo "Choose a fresh RESULT_DIR" >&2; exit 1; }
mkdir -p "$GROUP_DIR"
GROUP_DIR="$(cd -- "$GROUP_DIR" && pwd)"
cd "$REPO_ROOT"
# Check only selected datasets; no access to unselected paths is required.
"$PYTHON_BIN" - "$EVAL_SAMPLES" "$IMAGE_ROOT" "$SAMPLE_SEED" "$PROMPT_FIELD" "${stages[@]}" -- "${data_paths[@]}" <<'PY_CHECK' 2>&1 | tee "$GROUP_DIR/inspection.log"
import json
import sys
from pathlib import Path
sys.path.insert(0, 'tests')
from vlm_benchmark_data import read_eval_records
from reference_vlm_data import load_reference_records
boundary = sys.argv.index('--')
stages, paths = sys.argv[5:boundary], sys.argv[boundary+1:]
selected = dict(zip(stages, paths))
if 'multimodal_train' in selected and 'multimodal_val' in selected:
    assert Path(selected['multimodal_train']).resolve() != Path(selected['multimodal_val']).resolve(), 'train and validation manifests must differ'
for stage, path in zip(stages, paths):
    if stage not in {'text', 'multimodal_train', 'multimodal_val'}:
        rows, info = load_reference_records(path, int(sys.argv[1]), int(sys.argv[3]),
            sys.argv[2], stage.endswith('_vision'), sys.argv[4])
        print(json.dumps(dict(stage=stage, **info), ensure_ascii=False), flush=True)
    else:
        rows = list(read_eval_records(Path(path).expanduser(), int(sys.argv[1])))
        assert len(rows) == int(sys.argv[1]), f'not enough samples: {path}'
        assert all(bool(row.get('image')) == (stage != 'text') for row in rows), f'wrong input modality: {path}'
PY_CHECK
if [[ "${INSPECT_ONLY:-0}" == 1 ]]; then
    echo "Inspection: $GROUP_DIR/inspection.log"
    exit 0
fi
for index in "${!stages[@]}"; do
    stage="${stages[$index]}"; data="${data_paths[$index]}"; split="${splits[$index]}"
    echo "=== $stage: dataset=${dataset_names[$index]}, split=$split, data=$data, $AXIS, $EVAL_SAMPLES unique prompts per cell ==="
    REFERENCE_INPUT="${reference_inputs[$index]}" INPUT_MODE="${input_modes[$index]}" \
    EVAL_DATA="$data" EVAL_SPLIT="$split" EVAL_DATASET_NAME="${dataset_names[$index]}" RESULT_DIR="$GROUP_DIR/$stage" \
        bash tests/compare_vlm_16.sh 2>&1 | tee "$GROUP_DIR/$stage.log"
done
"$PYTHON_BIN" tests/report_text_vlm_speedup.py --root "$GROUP_DIR" --cases "${stages[@]}"
echo "Tables: $GROUP_DIR/speedup_tables.md"
echo "Full metrics: $GROUP_DIR/comparison.csv"
