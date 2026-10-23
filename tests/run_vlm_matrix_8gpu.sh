#!/usr/bin/env bash
# Seven existing drafts, six datasets, three batch sizes; one process per GPU.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export RESULT_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/vlm_matrix_$(date +%Y%m%d_%H%M%S)_$$}"
cd "$REPO_ROOT"
args=(--workspace "$DLITE_WORKSPACE" --output-dir "$RESULT_DIR")
read -r -a gpus <<< "${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
# Python accepts comma-separated or space-separated GPU identifiers.
args+=(--gpus "${gpus[@]}")
read -r -a batches <<< "${BATCH_SIZES:-1 16 64}"
read -r -a models <<< "${MODELS:-dlite_vlm_100k dflash_vlm_100k dflash_text_20k dlite_original dlite_vlm_20k dflash_vlm_20k dspark_vlm_20k}"
read -r -a tests <<< "${TESTS:-text-only charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k}"
read -r -a modes <<< "${STATE_MODES:-per_token compact_replay full_target_replay}"
args+=(--batch-sizes "${batches[@]}" --models "${models[@]}" --tests "${tests[@]}" --state-modes "${modes[@]}")
args+=(--samples "${EVAL_SAMPLES:-64}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}"
       --seed "${SAMPLE_SEED:-42}" --warmup "${WARMUP:-1}" --attention-backend "${ATTENTION_BACKEND:-auto}"
       --profile-samples "${PROFILE_SAMPLES:-16}")
[[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
[[ "${DRY_RUN:-0}" != 1 ]] || args+=(--dry-run)
[[ "${RETRY_OOM:-0}" != 1 ]] || args+=(--retry-oom)
[[ "${PROFILE_DECODE:-0}" != 1 ]] || args+=(--profile-decode)
exec "$PYTHON_BIN" tests/run_vlm_matrix.py "${args[@]}"
