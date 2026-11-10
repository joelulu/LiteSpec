# DFlash context and layer-adapter suites

Run the two launchers separately. Each trains its five models sequentially,
using all GPUs assigned to that suite for each model. Draft weights, optimizer
and schedule start fresh; the frozen Qwen3.5-4B Target loads normally.

| Suite | Models | Data | Epochs | GPUs per model | Micro | Accumulation | Global batch |
|---|---|---|---:|---:|---:|---:|---:|
| `bash tests/train_dflash_exp1.sh` | Full, Recent128, Pivot1, Pivot5, Pivot12 | existing 10w candidates | 5 | 8 | 4 | 8 | 256 |
| `bash tests/train_dflash_exp2.sh` | Pivot1, Pivot5, Pivot12, Pivot5-B, Pivot5-C | existing valid20k | 1 | 4 | 4 | 16 | 256 |

The second list has **five** models, as confirmed. B/C are modifications of
Pivot5, not a Cartesian product with the three baseline pivots.

Both keep block size 8, 512 training anchors, 5 draft layers, image budget 1024,
seed 42 and loss-decay gamma 4. Epoch exports are `epoch_1` through `epoch_N`,
with `last` pointing to final weights. Accumulation continues across epoch
boundaries, so an intermediate export occurs at the first completed optimizer
update after its epoch boundary. No optimizer/RNG training-state files are saved.
`MICRO_BATCH_SIZE=2` halves per-GPU micro batch and automatically doubles
accumulation while preserving global batch 256.

The presets replace stale epoch/data/checkpoint/debug/evaluation settings in the
environment. Optional `TRAIN_GPUS` is a comma-separated list of exactly 8 or 4
selectors. `SUITE_RUN_GROUP` can specify a fresh output path. Set `PYTHON_BIN`
explicitly if launching from a different active virtual environment.

## Layer adapters

Baseline Pivot5 uses the last visible Target position at selected layers
`[1,8,15,22,29]`, with one learned layer embedding per source and a shared H-to-H
input projection. Pivot12 keeps its existing 12-layer selection.

* **Pivot5-B** replaces the shared projection with five independent H-to-H
  projections, retaining the same five context slots and layer embeddings.
* **Pivot5-C** adds to B one learned `7 x 5` attention-score prior shared across
  draft layers and heads. Row `j-1` gives a source-layer bias for predicted block
  offset `j`, for `j=1..7`. Anchor offset 0 and draft-block keys receive no prior.
  This is a learned static preference per block position; content-dependent QK
  attention still operates normally. Zero initialization exactly recovers B.

All source slots at the same Target token repeat its original RoPE position.
The prior changes attention scores, not token positions or mask visibility.
Training uses a FlexAttention score modifier; SDPA/eager evaluation constructs
the equivalent additive bias. Unsupported attention backends fail explicitly.
The prior uses AdamW ownership; per-layer linear matrices keep normal optimizer
ownership. Saved `dflash_config` flags reconstruct the modules during loading.

Quick-suite startup runs `tests/check_dflash_layer_adapters.py --device cuda
--flex` before training. It checks shared/per-layer equivalence, zero-prior
equivalence, nonzero-prior block indexing and train/inference parity, optimizer
classification, real gradient updates, checkpoint reload, and dense/Flex
forward and gradient agreement. It loads no Target model or evaluation dataset.

## Preflight reuse

The long suite reuses
`runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb`.
This filters 95,420 candidates to **56,712 eligible records**. Five epochs use
these eligible records; the rejected 38,708 are not forced back into training.

The quick suite uses the existing
`data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl`, with
`MAX_SAMPLES=20000` and cache
`runs/dflash_context_20k_1ep_20261110_041837_1299222/full`.
Compatibility checks remain enabled; matching caches avoid image re-encoding.

## Automatic final evaluation

After all five successful training jobs, evaluate their `last` checkpoints
against same-GPU Target-only AR with frozen selections, **B=1 only**, 32 samples
per dataset (including both videos), seed 42, warmup 1 and max new tokens 256:

* GSM8K
* CharXiv, TextVQA, Flickr30k, HR-Bench 4K and 8K
* VideoDetailCaption and Video-MME (16 frames/video, 200704 pixels/frame)

Long-suite evaluation assigns the eight datasets to eight GPUs. Quick-suite
evaluation uses four independent GPU queues, one active dataset per GPU.
All checkpoints use the same frozen samples within a dataset. Video-MME retains
the existing video-description protocol, not official multiple-choice accuracy.
Token mismatches are reported alongside raw decode throughput ratios.

Evaluation sources retain the existing defaults; `DATA_DIR`, `IMAGE_ROOT`,
`TEXT_DATA` and `VIDEO_DATA_ROOT` can override their locations. The launcher
explicitly uses the existing reference prompt policy instead of inheriting
`PROMPT_FIELD=question`.

Outputs: `$RUN_GROUP/experiment.json`, per-model `train.log` and epoch weights,
`evaluate.log`, and `eval_all_b1_n32/{manifest.json,status.json,comparison.md,
comparison.csv,comparison.json}`. If a training job fails, stop before the next
training job or evaluation. Evaluation records individual dataset failures and
still runs the remaining datasets. Interrupts terminate only owned child jobs.

To repeat only evaluation after training, set `RUN_GROUP` to its output path and
pass all five explicit `--checkpoint LABEL PATH` entries to
`tests/evaluate_dflash_all.sh`; choose a fresh `RESULT_DIR` and set
`VIDEO_SAMPLES=32 EVAL_SAMPLES=32 EVAL_BATCH_SIZE=1`.
