# Replicated-input static batch benchmark

Run from LiteSpec after activating `.venv-vlm`:

```bash
EVAL_GPU=6 BATCH_SIZES="1 8" EVAL_SAMPLES=16 EVAL_MAX_NEW_TOKENS=128 \
    bash tests/compare_static_batch_text_vlm.sh
```

This defaults to three independent groups: the user's GSM8K text file, FineVision multimodal
`train.jsonl`, and FineVision multimodal `val.jsonl`. Checkpoint defaults match
`compare_vlm_16.sh`: AR, DLite 6/3 epochs, DFlash 3 epochs, DSpark 3 epochs, and
original Qwen3.5-4B-DLite. Dataset overrides are `TEXT_DATA`, `TRAIN_EVAL_DATA`,
`VAL_EVAL_DATA`; checkpoint overrides are `DLITE_6EP`, `DLITE_3EP`, `DFLASH_3EP`,
`DSPARK_3EP`, and `ORIGINAL_DLITE`.

Select models and groups with the same options as the round-robin wrapper:

```bash
EVAL_GPU=6 BATCH_SIZES="1 8" \
MODELS="ar dlite_original dspark_3ep" \
TESTS="text-only visual-eval visual-training" \
bash tests/compare_static_batch_text_vlm.sh
```

`MODELS` accepts `ar`, `dlite_original`, `dflash_3ep`, `dspark_3ep`, `dlite_3ep`,
`dlite_6ep` (space or comma separated). AR is always included for ratios;
`MODELS="ar"` is supported without any draft checkpoints. `TESTS` is space
separated; choose any subset of `text-only`, `visual-eval`, `visual-training`.
Only selected datasets and checkpoint files are required. One selected group
produces one table; all three produce three tables. Unselected inputs are not
read and unselected drafts are not loaded.

Default `TEXT_DATA` is
`$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main-0708-new/eval_datasets/gsm8k.jsonl`.
The evaluation reader accepts `{"turns": ["one complete prompt"]}` directly,
preserving newlines and the boxed-answer instruction. No conversion or file
rewrite occurs. Multi-turn or ambiguous question/turns records fail explicitly.
`TEXT_DATASET_NAME` defaults to `gsm8k`; `TEXT_SPLIT` defaults to `unspecified`
because the input file does not declare its official split. Both are recorded,
along with the manifest path. The normal image/question schema remains supported.

Every unique prompt is physically repeated along the batch dimension and all B
rows are forwarded together. Image patches and grid metadata are repeated in
matching order. Decoding is greedy; divergence between replicas raises an error.
With 16 prompts and B=8, each model processes 16 batches / 128 replicated
sequences, not 16 total sequences. Training answers are never fed as input.

This matches public LiteSpec's static replicated batch and decode-only timing
protocol. It is not continuous batching or a live serving system. Qwen3.5
multimodal prefill, MRoPE and hybrid state recovery are necessary adaptations to
the original Qwen3 text benchmark; its target cache cannot be used unchanged.

## Cache and timing

The target uses a native Transformers hybrid cache. Only full-attention layers
are replaced with fixed-capacity KV storage, allocated during prefill. Attention
receives a valid-prefix view; rejected suffixes are overwritten without copying
or reallocating attention history. On rejection the decoder snapshots recurrent
matrices, records/crops convolution history, crops KV, restores recurrence and
replays the accepted prefix. Pivot and next token use the replayed output.
Hybrid replay cost is included in decode time and remains a difference from
pure full-attention Qwen3 rollback.

Use the repo `.venv-vlm` Transformers hybrid-cache API. No dependencies are
installed or changed. `ATTENTION_BACKEND=auto` chooses FlashAttention 2 on CUDA
when `flash_attn` is installed, otherwise SDPA. Set `ATTENTION_BACKEND=sdpa` to
force SDPA. The same backend is selected for AR and speculative models. CPU
tests use float32 SDPA; real GPU runs use BF16. H100/FlashAttention performance
must be measured on the user's server.

Each B gets a short warmup for AR and each draft. CPU image processing, tensor
replication, loading and warmup are excluded. GPU prefill and initial draft
feature extraction are timed separately. Decode uses one synchronized wall-clock
interval including proposal, verification, recurrent snapshots, rollback/replay
and Python bookkeeping; no per-stage CUDA synchronizations are inserted.

Primary throughput is `sum(B * actual_output_length) / sum(decode_seconds)`.
The first generated token is counted as in the upstream default convention,
though it is predicted by prefill. Partial blocks/EOS count only emitted tokens.
Speedup divides each model's decode throughput by same-B, same-group AR decode
throughput. Each method uses its actual output-token count. Token equality is
recorded but does not gate observed ratios. Prefill-inclusive throughput is
also exported separately.

## Outputs

Results default to `runs/static_batch_text_vlm_TIMESTAMP_PID/`:

* `speedup_tables.md`: three tables, B columns, speedup / accepted length.
* `comparison.csv` / `comparison.json`: source split/path, batch size, unique
  prompts, completed sequences, decode throughput, prefill/decode times,
  prefill-inclusive throughput, latency and token matches.
* `text/`, `multimodal_train/`, `multimodal_val/`: settings, summaries and per-prompt
  `_bB.jsonl` records. Identical replicas are stored once with batch size and
  multiplied token counts.

Use `BATCH_SIZES="1 8 16 32 64"` for a larger sweep when memory permits. Completed
cells are written immediately; OOM/setup/runtime errors abort and preserve them.
Long-context scaling additionally requires controlled input-token length groups;
these datasets alone do not establish controlled long-context scaling. Split
labels identify source manifests, not proof of which records each draft saw.

The earlier `compare_text_vlm_16.sh` still defaults to the unbatched round-robin
reference. Do not mix its C columns or prefill-inclusive ratios with these B
columns or decode-only ratios in one plot.
