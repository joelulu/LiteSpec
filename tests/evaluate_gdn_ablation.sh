#!/usr/bin/env bash
# Also usable independently after an interrupted evaluation; RESUME=1 reuses cells.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
RUN_GROUP="${1:?Usage: bash tests/evaluate_gdn_ablation.sh RUN_GROUP}"
export RUN_GROUP
cd "$REPO_ROOT"
# Recover the training image budget instead of inheriting unrelated experiments.
IMAGE_MAX_TOKENS="$($PYTHON_BIN - "$RUN_GROUP" <<'PY'
from pathlib import Path
import json
import sys
root = Path(sys.argv[1])
configs = []
for arm in ('dflash', 'dlite', 'dflash_gdn', 'dlite_gdn'):
    path = root/arm
    if not (path/'training_complete.json').is_file():
        raise ValueError(f'{arm} did not finish training; inspect train.log')
    cfg = json.loads((path/'effective_config.json').read_text())
    configs.append(cfg)
for key in ('target_model', 'train_data', 'data_sha256', 'eligible_indices_sha256', 'image_max_tokens',
            'num_epochs', 'block_size', 'num_draft_layers', 'batch_size', 'accumulation_steps', 'world_size', 'seed'):
    if any(c.get(key) != configs[0].get(key) for c in configs[1:]):
        raise ValueError(f'four-arm training mismatch: {key}')
print(configs[0]['image_max_tokens'])
PY
)"
export IMAGE_MAX_TOKENS
export TARGET_MODEL="$($PYTHON_BIN - "$RUN_GROUP/dlite/effective_config.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['target_model'])
PY
)"
unset IMAGE_MIN_PIXELS IMAGE_MAX_PIXELS
RESULT_DIR="${RESULT_DIR:-$RUN_GROUP/evaluation}"
args=(--workspace "$DLITE_WORKSPACE" --output-dir "$RESULT_DIR" --gpus 4,5,6,7)
for arm in dflash dlite dflash_gdn dlite_gdn; do
    args+=(--checkpoint "$arm" "$RUN_GROUP/$arm/last")
done
read -r -a batches <<< "${BATCH_SIZES:-1}"
read -r -a modes <<< "${STATE_MODES:-per_token}"
read -r -a cases <<< "${TESTS:-text-only charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k}"
args+=(--batch-sizes "${batches[@]}" --state-modes "${modes[@]}" --tests "${cases[@]}"
       --samples "${EVAL_SAMPLES:-64}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}"
       --seed "${SAMPLE_SEED:-42}" --warmup "${WARMUP:-1}"
       --attention-backend "${ATTENTION_BACKEND:-auto}" --profile-samples "${PROFILE_SAMPLES:-16}")
[[ "${PROFILE_DECODE:-1}" != 1 ]] || args+=(--profile-decode)
[[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
[[ "${DRY_RUN:-0}" != 1 ]] || args+=(--dry-run)
[[ "${RETRY_OOM:-0}" != 1 ]] || args+=(--retry-oom)
exec "$PYTHON_BIN" tests/run_vlm_matrix.py "${args[@]}"
