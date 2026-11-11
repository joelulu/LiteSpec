# Simple long-context Full/Pivot5 experiment

Two entrypoints:

1. `bash tests/prepare_longctx_data.sh`: download/cache the MathVista testmini
   and GovReport test Parquet shards, then save **100** fixed single-image
   records and a pool of 50 report texts.
2. `bash tests/evaluate_dflash_longctx.sh`: take the **first 16** prepared records
   and evaluate short, 8192, 32768 and 65536 input-token variants, twice each.

Both use `PYTHON_BIN` (normally `$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python`).
Data preparation needs huggingface_hub, pandas, pyarrow and Pillow. It does not
need Torch/Transformers updates. The evaluator uses the existing working VLM
environment. Optional HF_ENDPOINT can point to the user's Hub mirror.

If the server cannot reach Hugging Face, download these two files on another
computer and upload them into `$DLITE_WORKSPACE/data/longctx_raw/`:

* MathVista: `https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/data/testmini-00000-of-00001-725687bf7a18d64b.parquet?download=true`
* GovReport: `https://huggingface.co/datasets/ccdv/govreport-summarization/resolve/main/document/test-00000-of-00001.parquet?download=true`

Then prepare from uploaded files, without any Hub access:

```bash
bash tests/prepare_longctx_data.sh \
  --mathvista-parquet "$DLITE_WORKSPACE/data/longctx_raw/testmini-00000-of-00001-725687bf7a18d64b.parquet" \
  --govreport-parquet "$DLITE_WORKSPACE/data/longctx_raw/test-00000-of-00001.parquet"
```

Keep the `.parquet` files intact; no extraction is needed. Local source paths
and SHA256 hashes are saved in the prepared manifest. Providing both parameters
avoids importing the Hub downloader. Existing prepared data is still verified
and reused.

## Sources and fixed choices

`longctx_reference.json` contains the first 100 MathVista IDs and 50 GovReport
row indices from the supplied portable_data archive, with its SHA256. The
download step uses the same source repositories and Parquet filenames as that
archive. A Parquet shard is downloaded as a unit; selecting 100 records does
not imply that only 100 rows cross the network. Only the 100 selected image
byte strings are written to the prepared image directory.

MathVista `query` is preserved as the one-turn `question`. The first five
selected records are printed/validated before publishing the data. The output
contains `base.jsonl`, `reports.jsonl`, `inspection_first5.json`, `images/` and
`manifest.json`. The manifest records dataset revisions, source shard hashes,
record IDs and hashes of every output file. Successful existing data is verified
and reused; incomplete/changed data is rejected rather than overwritten.

Defaults:

* Data: `$DLITE_WORKSPACE/data/longctx_mathvista100` (`LONGCTX_DATA_DIR`).
* Hub cache: `$DLITE_WORKSPACE/data/hf_longctx_cache` (`LONGCTX_HF_CACHE`).
* Models: Full/Pivot5 `epoch_5` exports from the most recent experiment-1
  `eval_all_b1_n32/manifest.json`. Each export must match the checkpoint in that
  evaluation. `SOURCE_EVAL_DIR` can specify its exact evaluation directory.
* Target: the exact local model fingerprint recorded by that evaluation.

## Length construction

Input order is image, report filler, original question. Short input keeps the
original image/question. Each sample has one deterministic long report sequence;
all longer variants use nested prefixes of its tokenizer IDs. The same 16 IDs,
image and original question are retained in every variant and repetition.

The **local Qwen3.5 processor** measures the entire encoded input, including
image expansion and chat overhead. Only filler text is adjusted to fit each
target, to within eight tokens below the budget. No image block or original
question is truncated. Per-image budget stays at 1024 visual tokens (converted
to pixel area using the model's own patch/merge sizes). No tokenizer from
Qwen3-VL or nominal portable-file lengths are used for this count.

Processor/target config loading and prompt preparation occur on CPU without
loading Target weights. `input_lengths.json` records actual lengths, visual
positions, filler cuts, report IDs and filler hashes. Configured Target and Draft
position limits must support the input plus 256 output tokens; no RoPE overrides
are applied.

## Execution and measurements

Before formal evaluation, two samples are tested at short and maximum length
with AR, Full and Pivot5. Any failure stops the formal sweep. On success the
eight jobs (four lengths times two repetitions) use GPU selectors 0 through 7
by default. Every job evaluates all three models sequentially on the same GPU.
Repetition 2 reverses the Full/Pivot5 order. `LONGCTX_GPUS` can change the
space-separated selectors, and `LONGCTX_RESULT_DIR` can specify fresh output.

Settings: **16 samples, B=1**, greedy, max_new_tokens=256, warmup=1, SDPA,
per_token state, correct images. The models are never retrained. Natural EOS
is preserved. Output lengths and AR matches remain visible; all samples
contribute to timing/acceptance metrics. A short-answer cell may be dominated
by prefill and first-draft work rather than steady decode.

Two samples per model/job receive a separate profiling pass after the normal
timed run. Profile-only additions split first/later drafting stages and record
the final projected Draft KV storage and raw hidden storage (including a view's
retained base storage). These are post-decode snapshots, not peak component
memory. The ordinary run reports total peak allocated GPU memory. Profiling
does not change ordinary generation or its measured times.

Outputs:

* `manifest.json`, `input_lengths.json`, frozen `inputs/*.jsonl`.
* `smoke/` and per-job `logs/`, `results/` with per-request details.
* `longctx_comparison.md`, `.csv`, `.json`: input/output lengths, committed
  length, first hit, tok/s, ratios to local AR/Full, prefill/decode/total times,
  peak memory and token matches. CSV/JSON additionally include first/later
  draft times, final cache/raw-hidden MiB and total-latency ratio to Full.

The main question is whether throughput relative to Full crosses 1 as input
length grows, and what acceptance/end-to-end latency tradeoff accompanies it.
These are controlled MathVista/GovReport probes extended to 100 source records,
not a byte-for-byte reproduction of the portable archive's 64-row prompts.

`--prepare-only` validates/tokenizes the 16 samples and freezes the experiment
without starting GPU jobs. This command creates an output directory; choose a
fresh `LONGCTX_RESULT_DIR` for a later benchmark run.

Local regression tests use stand-in raw records, tokenizers and model processes.
Actual Hub download, Qwen3.5 processor lengths and CUDA performance must be
verified in the training/evaluation environment.
