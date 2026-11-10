# DFlash Full/Pivot1 training commands recorded on 2026-11-10

## Recorded 10w command

The user's latest explicit command is **2 epochs, micro batch 4, accumulation
16**, Full on GPU 0/1/2/3 and Pivot1 on GPU 4/5/6/7. Each run's global batch is
256. The pasted command's old directory label `3ep_mb8_acc8` and printed
`3 epochs` message were stale labels; the environment variables were authoritative.
The recorded preset corrects those labels without changing these training settings.

```bash
bash tests/train_dflash_context_10w_pair.sh
```

| Setting | Recorded value |
| --- | --- |
| Target | `$HOME/LMM/lmm_model/Qwen3.5-4B` |
| Data | `data/finevision_qwen35_10w_img1024_relaxed/train.jsonl` |
| Candidate records read | 95,420 |
| Eligible / skipped | 56,712 / 38,708 |
| Sample limit | 0 (all candidates) |
| Initialization | From scratch |
| Validation | `runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb` |
| Contexts | Full and Pivot1 (time window 1) |
| Block / image budget | 8 / 1024 |
| Epochs / per-GPU micro batch / accumulation | 2 / 4 / 16 |
| GPUs | Four per run, two runs concurrently |
| Reporting / exports | TensorBoard / final inference weights only |

Paths are relative to `DLITE_WORKSPACE` unless specified otherwise. The preset
uses the sibling `LiteSpec/.venv-vlm/bin/python`, regardless of the active shell's
virtual environment. It reuses the completed preflight, using the compatibility
checks added in commit `8f99997`.

## New 20k weights-only adaptation

```bash
bash tests/train_dflash_context_20k_warmstart_pair.sh
```

Both runs load **the same original 10w Full DFlash weights**:

```text
runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dflash/last
```

The actual checkpoint directory is resolved (including a possible `last`
symlink), and its configuration hash is recorded in `effective_config.json`.
Full keeps Full context; Pivot1 switches visibility to the single pivot. All
backbone/projector parameters must load exactly: missing, unexpected or
mismatched parameters fail. This is weights-only adaptation with a new optimizer
and a new two-epoch schedule, not resumption of the old training state.

Both runs use `data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl`,
`MAX_SAMPLES=20000`, 2 epochs, per-GPU micro batch 4, accumulation 16, image budget
1024 and block size 8. Their GPU assignments and reporting match the recorded
10w preset. The reusable 20k preflight is:

```text
runs/dflash_context_20k_1ep_20261110_041837_1299222/full
```

This cache has a recorded 20,000 eligible records and 0 skipped; its original
one-epoch setting does not constrain the new training epoch count. The 10w
validation cache must not be substituted for it. Override `INIT_CHECKPOINT` or
`VALIDATION_FROM` explicitly if using a different compatible source/cache.

The pair launcher writes a new output directory with the real epoch/batch
settings in its name, and prints the exact `tail -f` command and final checkpoint
paths. Stop older training jobs using the same GPUs before launching a new pair.

## From scratch on 10w, saving each epoch

```bash
SAVE_EVERY_EPOCH=1 bash tests/train_dflash_context_10w_pair.sh
```

This is the recorded 10w pair (two epochs, four GPUs per mode, micro batch 4,
accumulation 16) with no draft initialization checkpoint. The frozen Target
still loads its normal Qwen3.5-4B weights. Each mode exports `epoch_1`, `epoch_2`
and `last`. Epoch directories and `last` are atomic symlinks to complete
`step_XXXXXXX` weight exports, so an epoch/final boundary does not duplicate the
same model. Optimizer/RNG training-state files remain disabled.

As in the existing runner, an epoch export occurs at the first completed
optimizer update at or after its boundary; accumulation carries across epochs.
Partial gradients are neither discarded nor applied early just to save a model.
The default `SAVE_EVERY_EPOCH=0` retains final-only behavior for older commands.
