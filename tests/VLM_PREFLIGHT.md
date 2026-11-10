# Parallel VLM preprocessing

Run from the DFlash context worktree. The wrapper selects the HF VLM Python,
uses eight CPU threads (one independent processor per thread), and does not load
Target weights. Keep a GPU visible: importing the training dependency `yunchang`
queries its architecture and may initialize a small CUDA context. This is not a
multi-GPU preprocessing job.

## Preprocess independently

```bash
unset REUSE_VALIDATION_FROM CONFIG
VALIDATION_WORKERS=8 bash tests/prepare_vlm_validation.sh
```

Defaults: `finevision_qwen35_10w_img1024_relaxed/train.jsonl`, all records
(`MAX_SAMPLES=0`), block size 8, image budget 1024. Override `DLITE_WORKSPACE`,
`PYTHON_BIN`, `TRAIN_DATA`, `TARGET_MODEL`, `IMAGE_MAX_TOKENS`, or `MAX_SAMPLES`
as needed. Set `CONFIG` to an existing training YAML to use its exact settings;
otherwise the wrapper generates the Full DFlash context configuration.
When `CONFIG` is supplied, its image budget takes precedence over the environment.
`PRECHECK_CUDA_VISIBLE_DEVICES` may select the one GPU available during import.

New caches are saved under `$DLITE_WORKSPACE/runs/vlm_validation/`. The helper
also searches older completed runs for compatible caches. A successful run prints
`export REUSE_VALIDATION_FROM=...`; newly built caches include
`reuse_validation.env`, which can be sourced before running the original training
command. An explicit `REUSE_VALIDATION_FROM` is checked strictly: incompatible
caches fail instead of being silently accepted. Unset it to search/build again.

## Automatically preprocess before training

`tests/train_dflash_context.sh` now enables preprocessing by default. The existing
20k defaults of that training script remain in effect; it passes its actual YAML,
data path and sample limit to preprocessing. Example for Full/Pivot1 on 10w data:

```bash
unset REUSE_VALIDATION_FROM
TRAIN_DATA="$DLITE_WORKSPACE/data/finevision_qwen35_10w_img1024_relaxed/train.jsonl" \
TRAIN_SAMPLES=0 EPOCHS=3 MICRO_BATCH_SIZE=8 ACCUMULATION_STEPS=8 \
CONTEXTS="full pivot1" VALIDATION_WORKERS=8 \
bash tests/train_dflash_context.sh
```

That entrypoint trains modes sequentially using its existing GPU settings.
For custom launchers (including concurrent four-GPU Full/Pivot1 jobs), export
`PREPARE_VALIDATION=1` and `VALIDATION_WORKERS=8` before invoking
`scripts/dlite/run_training_vlm.sh`. It runs preprocessing before loading Target
weights and supplies the resulting cache to training automatically. Use environment
variables and `CONFIG` for overrides; this automatic hook rejects additional CLI
arguments rather than preprocessing a different configuration from the trainer.
Other VLM launchers retain their original behavior unless explicitly opted in.

Concurrent jobs with the same eligibility settings share a file lock. Only one
job checks images; others wait and reuse its complete result. Completed caches
survive subsequent training failures. Interrupted checks are not resumable, and
incomplete caches are never published. Use `PREPARE_VALIDATION=0` to retain the
original training preflight path.

## Compatibility and limitations

Serial and parallel preprocessing call the same `validate_record` rule, including
the optional two-round anchor constraint. Results are sorted by original record
index; the trainer's own `reuse_validation` verifies IDs, counts and eligible-set
hashes before publication. Data hash/path, Target path, maximum length, block size,
sample limit, image budget and two-round mode must match. Batch size, epoch count,
GPU count and Full/Pivot context choice do not change this eligibility set.

Images and processor files must remain unchanged, as with the original reuse
mechanism: the JSONL hash does not hash image contents or processor files. After
changing those files, unset `REUSE_VALIDATION_FROM` and run
`bash tests/prepare_vlm_validation.sh --force`; use its printed directory for
training. Full/Pivot1 DFlash weights-only initialization is resolved before the
eligibility check; see [recorded training presets](DFLASH_CONTEXT_RUNS.md).
Reader-only adaptation and other init-checkpoint modes are not supported by this
helper; use their original runner's preflight.

Increase `VALIDATION_WORKERS` only when CPU/storage capacity allows it; speedup
depends on image I/O and preprocessing. `VALIDATION_CHUNK_SIZE` defaults to 64.
All rejected records retain their original index, ID and reason in
`skipped_training.jsonl`. Unexpected I/O or processing errors fail the check;
they are not silently treated as rejected data.
