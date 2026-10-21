# AR and original DLite on five local vision datasets

## Unified selectable static-batch entry point

The five logical names resolve to these source files. Flickr30k and TextVQA
prefer the explicit `_test` split; legacy unsuffixed files are accepted only
when the corresponding test file is absent. Training files are never a fallback.

| Selector | Filename |
|---|---|
| `charxiv_val` | `charxiv_val.jsonl` |
| `flickr30k` | `flickr30k_test.jsonl` |
| `textvqa` | `textvqa_test.jsonl` |
| `hr_bench_4k` | `hr_bench_4k.jsonl` |
| `hr_bench_8k` | `hr_bench_8k.jsonl` |

Use the existing entry point to compare any subset of the six checkpoints:

```bash
EVAL_GPU=6 BATCH_SIZES="1 8" EVAL_SAMPLES=16 EVAL_MAX_NEW_TOKENS=128 \
MODELS="ar dlite_original dflash_3ep dspark_3ep dlite_3ep dlite_6ep" \
TESTS="charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k" \
bash tests/compare_static_batch_text_vlm.sh
```

Named datasets use images by default. Set `MODES="text vision"` to include
same-question image-free controls. Existing `text-only` (GSM8K), `visual-eval`
(FineVision val), and `visual-training` (FineVision train) retain their meanings
and may be mixed with dataset names in `TESTS`.

Alternatively, use a dataset group and select a subset through `DATASETS`:

```bash
EVAL_GPU=6 BATCH_SIZES="1 8" EVAL_SAMPLES=16 EVAL_MAX_NEW_TOKENS=128 \
MODELS="ar dlite_original dlite_6ep" \
TESTS="text-only visual-eval visual-training visual-datasets" \
DATASETS="charxiv_val textvqa" \
bash tests/compare_static_batch_text_vlm.sh
```

`DATASETS` applies only to `visual-datasets`; without it that group expands to
all five named datasets. `TESTS` never implicitly changes FineVision's data.
Only selected files are inspected. `INSPECT_ONLY=1` runs the preflight without
loading models. Combined results use the existing `speedup_tables.md` and
`comparison.csv/json` under `runs/static_batch_text_vlm_<timestamp>_<pid>/`.

`MODELS=ar` also works. AR is always included as the speedup baseline. Existing
`DLITE_6EP`, `DLITE_3EP`, `DFLASH_3EP`, `DSPARK_3EP`, and `ORIGINAL_DLITE` path
overrides are forwarded unchanged.

## Five-dataset convenience entry point

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
`ar dlite_original` (overridable with `MODELS`), B=1, 16 unique prompts per dataset, seed 42,
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
