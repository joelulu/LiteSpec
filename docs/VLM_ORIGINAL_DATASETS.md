# AR and original DLite on five local vision datasets

Run from `LiteSpec/`. This is the HF static replicated-batch benchmark,
not a SGLang service or continuous batching benchmark. Both decoders use the
same Qwen3.5 target. Each unique prompt is replicated B times in one forward.

```bash
git pull --ff-only origin research/qwen35-vlm
EVAL_GPU=6 MODES="vision" BATCH_SIZES="1" \
SAMPLES_PER_DATASET=16 MAX_NEW_TOKENS=512 \
bash tests/compare_original_vlm_datasets.sh
```

Defaults: `DATA_DIR=$HOME/LMM/liyt/datasets/jsonl_datas`,
`IMAGE_ROOT=$DATA_DIR/imgs`, datasets
`charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k`, models
`ar dlite_original`, B=1, 16 unique prompts per dataset, seed 42,
512 maximum generated tokens, greedy decoding, thinking disabled.
Without `MODES`, both `text vision` run. `text` is the same task question
with its image removed; it is not a separate text benchmark such as GSM8K.

The reference prompt policy uses nonempty `prompt`, otherwise the first
`conversations[0].content`, matching the supplied evaluation script. No source
files are converted or rewritten; answers are never included in prompts.
One local image path per record is required for vision. Absolute paths work;
relative paths resolve against `IMAGE_ROOT`, with duplicate `imgs/` removed.
An explicit `PROMPT_FIELD=question`, `prompt`, or `first-content` can override
the reference policy. Invalid records or missing images stop the run.

Inspect the real data without loading weights:

```bash
INSPECT_ONLY=1 MODES="vision" bash tests/compare_original_vlm_datasets.sh
```

Preflight prints each source's first five records (prompt preview, field used,
resolved image path and original W/H), the count and sampled source rows.
Sampling matches `random.Random(42).sample` in the supplied script; text and
vision use the same rows. Fewer than 16 available records means all records.
Set `SAMPLES_PER_DATASET=0` for all records.

Optional subset, static batch sizes, and image pixel-area budget:

```bash
EVAL_GPU=6 DATASETS="textvqa hr_bench_4k" MODES="text vision" \
BATCH_SIZES="1 8" SAMPLES_PER_DATASET=16 MAX_NEW_TOKENS=512 \
IMAGE_MAX_PIXELS=1048576 \
bash tests/compare_original_vlm_datasets.sh
```

`IMAGE_MIN_PIXELS` and `IMAGE_MAX_PIXELS` are pixel **areas**, applied to the
active Qwen processor size policy. When omitted, the model processor's defaults
are retained. HRBench's source 4K/8K resolution does not guarantee that full
resolution reaches the model: `inputs.json` records original W/H, processed H/W,
grid T/H/W, actual image-token counts and total prompt tokens. Start at B=1;
large images and batches need more GPU memory. Paths and limits are recorded in
`settings.json`. Override `TARGET_MODEL`, `ORIGINAL_DLITE`, or `PYTHON_BIN` as needed.

Results are under `$DLITE_WORKSPACE/runs/original_vlm_datasets_<timestamp>_<pid>/`:

- `inspection.json`: actual source inspection and sampling metadata.
- `<dataset>/<mode>/inputs.json`: processed image/token metadata.
- `<dataset>/<mode>/{ar,dlite_original}_b<B>.jsonl`: generated text, token IDs,
  acceptance lengths, timings and AR token-match flags.
- `comparison.md`, `.csv`, `.json`: combined tables and metrics.

The main speedup is DLite decode tokens/s divided by AR decode tokens/s at the
same B; prefill is excluded. CSV/JSON also contain prefill time, end-to-end
throughput ratio and peak allocated GPU memory. Acceptance length includes the
guaranteed-progress token. Token match is recorded but does not gate ratios.
These outputs measure decoding performance and preserve predictions for review;
they do not implement dataset-specific task accuracy scoring.
