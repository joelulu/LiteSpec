# Prompt memory ablation for history-free DLite

Branch: `research/qwen35-vlm-memory`. Training remains direct SFT, not the
two-stage teacher/student recipe. No target weights or GDN state kernels change.

## Run

From `LiteSpec/`, in the working HF VLM environment:

```bash
unset MAX_STEPS MAX_SAMPLES CONFIG OUTPUT_DIR ALGORITHM REUSE_VALIDATION_FROM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
EPOCHS=3 MICRO_BATCH_SIZE=2 ACCUMULATION_STEPS=4 BLOCK_SIZE=8 \
IMAGE_MAX_TOKENS=1024 MEMORY_MAX_TOKENS=1024 MEMORY_RANK=256 \
EVAL_GPU=0 EVAL_SAMPLES=64 EVAL_MAX_NEW_TOKENS=256 BATCH_SIZES=1 \
bash tests/train_memory_ablation.sh
```

`TRAIN_DATA` defaults to the existing
`finevision_qwen35_10w_img1024_relaxed/train.jsonl`. This is a 100k-candidate
corpus, not a promise of 100k eligible examples. Existing rollout answers have
a 256-token generation cap; no answers are regenerated or extended here.
Eight GPUs give effective batch 64. Three modes share seed, backbone/head
initialization, sample ordering, optimizer schedule and eligible examples.
Memory initialization preserves the RNG stream. Target is frozen; backbone,
head and reader are jointly trained from scratch, matching the trained baseline.
Do not initialize one arm from an existing trained checkpoint.

The first arm performs image input preflight once. The next two reuse its
eligible set after compatibility checks (including image token cap). An old
preflight without explicit image budget metadata is deliberately not reused.
TensorBoard/ETA are inherited. `memory/source_tokens` records actual memory
length during training. Configuration, eligible counts and checkpoints are
saved per mode under `RUN_GROUP/{none,vision,text}/`; final links are `last/`.

For a short server smoke run before the full job:

```bash
DEBUG_RUN=1 MAX_SAMPLES=32 MAX_STEPS=1 RUN_EVAL=0 \
bash tests/train_memory_ablation.sh
```

Provide enough eligible records for all GPUs/micro batches. Debug artifacts
must not be used as the actual three-epoch comparison. `RUN_EVAL=0` skips
evaluation. `TESTS="charxiv_val textvqa"` restricts evaluation. Evaluation-only:

```bash
EVAL_GPU=0 EVAL_SAMPLES=64 EVAL_MAX_NEW_TOKENS=256 BATCH_SIZES=1 \
python tests/evaluate_memory_ablation.py --run-group /absolute/path/to/run_group
```

## Architecture and causality

- Modes `none`, `vision`, `text` are checkpoint configuration, not inferred
  from a model label. Old checkpoints default to `none` and keep the old path.
- Select target layers at the first, middle and last configured CHS indices;
  average them at each selected source position. Exact layer IDs are serialized.
  This small initial reader is an explicit design choice, not an information
  upper bound over all possible visual readers.
- Vision reads image payload positions only, excluding delimiters. Text reads
  nonvisual prompt positions, including chat-template text; no answer positions
  or padding. Each training sample carries its own prompt boundary.
- Preserve up to 1024 source positions. Longer sources are sampled at evenly
  spaced indices, without learned compression or manual image resizing.
  The processor's image pixel cap is computed from patch size × merge size.
- Pivot layers are averaged to form one query per anchor. A rank-256 SDPA reader
  produces a gated residual added to every Draft query position in that block.
  Original CHS conditioning, masks, local positions and prediction head remain.
- Training projects each example's memory once for all sampled anchors; no
  `[anchors × image_tokens × hidden]` memory replication. Inference caches
  projected K/V once at prefill and reads once per speculative round. This is a
  fixed-size auxiliary memory, **not zero extra KV**, and does not accumulate
  generated history.
- A missing image gives an exactly zero visual-reader residual, including on
  GSM8K. Both trained backbones may still differ because their training differs.

Text hidden states can already contain image information. The text arm controls
the **source positions** and reader parameters, not visual information removal.
Identical maximum capacity and parameter count are provided; actual text/image
token counts differ and are reported. This is not a strict equal-information
or equal-FLOP comparison. Reducing `MEMORY_MAX_TOKENS` later is a separate sweep.

## Evaluation and diagnosis

Default: fixed seeded 64 prompts each from GSM8K, CharXiv, Flickr30K, TextVQA,
HR-Bench 4K/8K; generation cap 256; B=1; `per_token` state management. Reuses
the existing matrix dataset selection and snapshots exact normalized prompts.
All arms use the same image cap as training. This is static replicated batch,
not online continuous batching. No task accuracy grader is added.

Each dataset measures AR and all three trained models. For vision datasets,
another pass evaluates the **same vision checkpoint** with other-image memory:
fixed seed, a different image path, original question retained. The primary
target prefill, its correct image, pivot and target verification remain intact.
At least two distinct image paths are required. This is a diagnostic, not a
fourth trained model. `EVAL_DISABLED=1` also disables the same reader at inference.
Checkpoint parameters are not changed by either diagnostic.

Wrong-image memory needs an extra donor target prefill in this HF implementation.
It is included in reported prefill/end-to-end time and `donor_prefill_seconds`;
do not compare that end-to-end rate as a deployable method. Decode-only time
includes actual memory reads. Correct-memory construction/projection is counted
in prefill and `memory_build_seconds`. `PROFILE_DECODE=1` enables separate
instrumented passes; stages `memory_read` and `draft_backbone` are separated.
Other existing draft stages remain unchanged. Token match is retained in JSON,
while raw throughput ratios follow the existing benchmark convention.

Reports: `RUN_GROUP/evaluation/memory_tables.md`, `comparison.csv`, `summary.json`,
sample snapshots, donor mappings, per-request outputs and detailed per-cell logs.
GSM8K does not run wrong-image diagnosis. Use `--output-dir` for a fresh evaluation.

## Validation scope

CPU numerical tests cover answer/padding exclusion, empty modality, projected
cache equivalence, RNG/backbone initialization, loss/backprop, optimizer ownership,
checkpoint roundtrip, batched real hybrid target inference and wrong-image AR
baseline reuse. Actual H100 FSDP training, BF16 kernels, memory consumption and
speed require the short server run; this development environment has no GPU.
Memory checkpoints use the supplied static-batch evaluation entrypoint. Legacy
`spec_generate`/old smoke entrypoints do not construct memory and fail explicitly
rather than silently evaluating the checkpoint without its reader. SGLang/vLLM
deployment is outside this branch's scope.
