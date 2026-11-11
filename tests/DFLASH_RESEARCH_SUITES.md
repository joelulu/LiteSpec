# DFlash context and layer-adapter suites

Run the two launchers separately. Each trains its models sequentially,
using all GPUs assigned to that suite for each model. Draft weights, optimizer
and schedule start fresh; the frozen Qwen3.5-4B Target loads normally.

| Suite | Models | Data | Epochs | GPUs per model | Micro | Accumulation | Global batch |
|---|---|---|---:|---:|---:|---:|---:|
| `bash tests/train_dflash_exp1.sh` | Full, Recent128, Pivot1, Pivot5, Pivot12 | existing 10w candidates | 5 | 8 | 4 | 8 | 256 |
| `bash tests/train_dflash_exp2.sh` | Pivot5, Pivot5-B, Pivot5-C | existing 10w candidates | 5 | 4 | 4 | 16 | 256 |

The second suite trains Pivot5 as a matched four-GPU control alongside its B/C
adapters. Full, Recent128, Pivot1 and Pivot12 remain covered by experiment 1.

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

Adapter-suite startup runs `tests/check_dflash_layer_adapters.py --device cuda
--flex` before training. It checks shared/per-layer equivalence, zero-prior
equivalence, nonzero-prior block indexing and train/inference parity, optimizer
classification, real gradient updates, checkpoint reload, and dense/Flex
forward and gradient agreement. It loads no Target model or evaluation dataset.

The suite also runs a four-rank check with `--fsdp`, using the production
SHARD_GRAD_OP/use_orig_params setting, BF16 weights, compiled TRITON FlexAttention,
the FP32-master BF16Optimizer, and two optimizer updates with 16 accumulation
passes each. Sequence lengths vary while gradients are retained. The learned
prior must receive a finite nonzero gradient and change after optimization.

### C capture failure and recovery

In torch 2.9.1, a score-modifier closure capturing the original FSDP bias view
can make Dynamo recursively inspect a full flat-parameter buffer whose gradient
is a local shard. The resulting error mentions assigning a gradient whose size
is one quarter of the tensor size on four GPUs. Single-GPU checks cannot cover
this state. The model now clones the 7-by-5 prior once before entering compiled
attention, sharing the copy across draft layers. CloneBackward keeps the prior
trainable; neither detach nor dense-training fallback is used.

If Pivot5/B finished and C failed, retain their exports and run:

```bash
FAILED_RUN_GROUP=/absolute/path/to/the/failed/experiment/group \
TRAIN_GPUS=0,1,2,3 bash tests/retry_dflash_exp2_c.sh
```

The retry launcher checks both completed checkpoints, runs single-GPU and
four-GPU adapter checks, then starts C **from scratch** with the same 10w data,
five epochs and global batch 256. It creates a fresh `pivot5_C_retry_*` directory,
preserving the previous C logs and any exports. On success it evaluates the old
Pivot5, old B and new C at B=1/n=32 across all eight datasets, writing the report
under the retry directory's `eval_all_b1_n32`. If the distributed check fails,
neither long training nor evaluation starts.

Local checks without Torch can exercise the capture boundary and launcher
orchestration. The real Tensor/FakeTensor checks in
`tests/test_dflash_layer_prior_fsdp.py` require Torch; the distributed regression
requires the training environment and multiple GPUs.

## Preflight reuse

Both suites use `data/finevision_qwen35_10w_img1024_relaxed/train.jsonl`,
with `MAX_SAMPLES=0`, and reuse
`runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb`.
This filters 95,420 candidates to **56,712 eligible records**. Five epochs use
these eligible records; the rejected 38,708 are not forced back into training.

Compatibility checks remain enabled; matching caches avoid image re-encoding.

## Automatic final evaluation

After all training jobs in the suite succeed, evaluate their `last` checkpoints
against same-GPU Target-only AR with frozen selections, **B=1 only**, 32 samples
per dataset (including both videos), seed 42, warmup 1 and max new tokens 256:

* GSM8K
* CharXiv, TextVQA, Flickr30k, HR-Bench 4K and 8K
* VideoDetailCaption and Video-MME (16 frames/video, 200704 pixels/frame)

Long-suite evaluation assigns the eight datasets to eight GPUs. Adapter-suite
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
pass the suite's explicit `--checkpoint LABEL PATH` entries (five for experiment
1, three for experiment 2) to
`tests/evaluate_dflash_all.sh`; choose a fresh `RESULT_DIR` and set
`VIDEO_SAMPLES=32 EVAL_SAMPLES=32 EVAL_BATCH_SIZE=1`.
