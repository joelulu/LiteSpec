# Sixteen-prompt checkpoint comparison

For true replicated-input static batches with decode-only timing, use
`tests/compare_static_batch_text_vlm.sh`; see [the static batch guide](VLM_STATIC_BATCH.md).
The C-based round-robin tests below remain available as reference diagnostics.

`tests/compare_vlm_16.sh` defaults to the three supplied checkpoints:

* `../runs/dlite_vlm_aligned_6ep_20261021_022447/last`
* `../runs/vlm_three_3ep_20261021_045040_2983971/dlite/last`
* `../runs/vlm_three_3ep_20261021_045040_2983971/dflash/last`

It runs AR, DLite 6ep, DLite 3ep and DFlash 3ep on the exact same first 16
validation records, generating at most 128 tokens each, with one warmup request
per model and concurrency levels 1, 4 and 8. Models run sequentially on the same
GPU (default physical GPU 0); there is one shared target and at most one loaded
draft. Stop other workloads on this GPU before comparing timing.

```bash
bash tests/compare_vlm_16.sh
# Fast single-request check first:
CONCURRENCIES=1 bash tests/compare_vlm_16.sh
# Correctness diagnostic, whose timings must not be used as speed benchmarks:
VERIFICATION=sequential bash tests/compare_vlm_16.sh
# Add the original local checkpoint to the same comparison:
ORIGINAL_DLITE="$HOME/LMM/lmm_model/Qwen3.5-4B-DLite" \
CONCURRENCIES=1 bash tests/compare_vlm_16.sh
# Compare only AR and the original checkpoint:
ONLY_ORIGINAL=1 CONCURRENCIES=1 bash tests/compare_vlm_16.sh
```

The shell script may also be copied to the parent `dlite-vlm` directory. Override
`DLITE_6EP`, `DLITE_3EP`, `DFLASH_3EP`, `EVAL_DATA`, `EVAL_GPU`, `EVAL_SAMPLES`,
`EVAL_MAX_NEW_TOKENS`, `CONCURRENCIES`, `WARMUP` or `RESULT_DIR` explicitly.
Use a fresh RESULT_DIR. The default data is
`../data/finevision_qwen35_sglang_20000/val.jsonl`.

## What concurrency means here

This is a **single-GPU, round-robin, unbatched HF reference scheduler**. Up to C
requests keep independent target hybrid caches, draft caches and pivot features
alive. Each request advances through prefill or one decode/verification round,
then yields; another request runs next. MRoPE deltas stored on the shared Qwen3.5
model are explicitly saved/restored per request. Threads sharing a raw HF model
would not provide this isolation. Requests are not distributed over multiple GPUs
and target weights are not duplicated C times.

**There is no batched forward, continuous batching, GPU-kernel overlap or
production serving backend.** The experiment measures cache coexistence and
request interleaving in this reference implementation. It cannot establish the
high-concurrency throughput advantage of DLite over DFlash in SGLang/vLLM.
Use serving-engine integration for that claim. Throughput may stay flat or fall
as C increases; that is not evidence about an optimized serving scheduler.

## Metrics and timing

* `mean_committed_length`: global mean tokens committed per speculative round,
  including its guaranteed token; `mean_accepted_draft_tokens` subtracts one.
  Means are weighted by number of rounds, not averaged equally over requests.
* `output_tokens_per_second`: sum of actual generated output token counts across
  all 16 requests divided by the **whole cell's wall time**. It includes model
  prefill, drafting, verification, full hybrid-cache snapshots/replay and reference
  scheduler overhead. EOS tokens are counted if emitted. Model loading, warmup,
  CPU image preprocessing and initial GPU input transfer are outside the timer.
* Mean/P95 latency: admission into the active pool through completion. Requests
  beyond C are admitted when a slot frees (closed loop); latency includes time
  spent waiting while other active requests advance. No artificial sleep is used.
* `speedup_vs_ar`: AR cell wall time / draft cell wall time at the same C, only
  valid after exact token equality and only for block verification. The AR
  reference is also checked for consistency between concurrency levels.
* Peak allocated GiB: total allocated GPU memory including model weights, retained
  input tensors and request caches; not KV-only memory or allocator reserved bytes.
* Profile totals: `prefill_seconds`, `draft_seconds`,
  `verify_and_snapshot_seconds`, and `recovery_seconds` sum existing per-request
  stage timers. Verification includes copying the hybrid cache; replay is separate.
  `other_seconds` is the remaining wall time (including AR decode for the AR row).
  These fields do not add new CUDA synchronization points.

Each run writes `settings.json` (checkpoint paths and exact sample IDs),
`summary.csv`, `summary.json`, and one JSONL per model/concurrency with output
tokens, acceptance lengths, timing and match flags. The terminal prints a compact
summary, mismatch sample IDs, and per-request first divergent token positions.
The original checkpoint retains its own architecture version when loaded; it is
not converted into the locally trained model's version.
Any mismatch makes the final command fail and invalidates the affected
speedup; remaining comparisons are still collected. Exceptions such as OOM abort
the run and leave already written results intact.

Block verification on a BF16 hybrid target can differ numerically from tokenwise
AR even with correct cache recovery. The existing sequential diagnostic remains
available, but its timings do not measure speculative acceleration. Sixteen
samples are a smoke check, not a stable statistical benchmark. Short outputs and
the final partially filled block can affect observed acceptance lengths.

The prefill path now projects only its last logit position and releases unneeded
all-layer prefill outputs before yielding. DLite copies the small pivot slice so
it does not accidentally retain a full prompt-length hidden-state tensor.

CPU tests use real tiny hybrid Qwen3.5 targets to compare interleaved image/text
requests with isolated execution, as well as fake-generator accounting and a
complete CLI export test. Real H100 checkpoints/data and GPU concurrency timing
must be verified on the user's server.

## Text first, then multimodal diagnostics

Run `EVAL_GPU=0 bash tests/diagnose_dlite_text_then_vlm.sh` from LiteSpec.
It uses the original checkpoint at `$HOME/LMM/lmm_model/Qwen3.5-4B-DLite`.
Override `ORIGINAL_DLITE`, `TARGET_MODEL`, `PYTHON_BIN`, `TEXT_DATA`, `EVAL_DATA`,
`EVAL_SAMPLES` (default 16), or `EVAL_MAX_NEW_TOKENS` (default 128) if needed.

The first stage compares AR with the original DLite `spec_generate` text entry.
This uses the original recurrent-state transaction/KV crop path rather than the
VLM reference's full hybrid-cache copies and per-stage CUDA synchronization.
It keeps the checkpoint's architecture version and greedy decoding. This remains
an HF single-request test, not an optimized serving-engine benchmark.
Native AR uses the same target, prompt and output cap with a cached token loop.
Native total wall time includes prefill and decode. Detailed stage times are
marked unavailable, rather than adding synchronization to this timing control.

The second stage runs the same text prompts through the VLM reference path.
Both reference stages also compare the trained DSpark checkpoint, defaulting to
`runs/vlm_three_3ep_20261021_045040_2983971/dspark/last` under the workspace;
override it with `DSPARK_3EP`. DSpark uses its own draft implementation and cache
through the shared HF reference scheduler. The native text stage remains DLite
only because `spec_generate` there is a DLite-specific entry point. DSpark token
matches and timing are reported independently; they do not change the DLite
text gate. The general `tests/compare_vlm_16.sh` also includes DSpark by default.
This isolates the effect of changing the execution path with fixed prompts.
Only if both text paths match their respective AR tokens exactly and the native
text path has wall-time speedup greater than one does the script start image
evaluation. Otherwise it stops with the text results retained. Set
`RUN_MULTIMODAL_ALWAYS=1` only to explicitly collect image diagnostics anyway.

The image stage uses the existing val manifest and reports prefill, drafting,
verification plus cache snapshot, rejected-prefix replay, and residual wall time.
The same breakdown for text reference helps identify reference-runtime overhead.
Text prompts are hand-authored smoke prompts in `tests/data/text_smoke_16.jsonl`;
they are not image questions with images removed. Text and image prompt sets
differ, so their total-time difference cannot be attributed entirely to vision.
Each stage has independent warmup, AR correctness comparison, JSONL token
details, and CSV/JSON summaries. Group results are written under
`runs/dlite_text_then_vlm_*/analysis.csv` and `analysis.json` with stage logs.

Sequential verification can establish tokenwise correctness but cannot measure
speculative acceleration: it still runs the target once per verified token.
Block-only mismatches require first-divergent-token/cache investigation before
attributing them to numerics or claiming valid speedups. A short smoke-test
speedup near one should be repeated with more prompts and longer outputs.

## Locate a first token mismatch

Use `tests/diagnose_vlm_mismatch.py --previous-run GROUP_DIR --output-dir NEW_DIR`
with the group directory from `diagnose_dlite_text_then_vlm.sh`. It selects the
first failed text request (override with `--sample-id text03`) and repeats native
block, native sequential, and reference block generation. It reuses the recorded
target/checkpoint paths and output cap, not a new training run.

For each cached multi-token target call it clones the incoming cache, compares
the actual chunk computation with one-token calls from that same snapshot, and
records token predictions, logit differences, sequential top-two margins, and
attention/conv/recurrent cache tensor differences. It restores model-level MRoPE
state and returns the original call's output unchanged. See `diagnosis.json` and
`*_target_audit.jsonl`. All timing from this audit is invalid for benchmarking.

A local chunk-vs-token prediction disagreement establishes an execution-path
difference with equal starting cache/input, but does not by itself prove harmless
floating-point error or rule out a kernel bug. Small logit margins support the
numerical-sensitivity hypothesis. Cache differences can also be numerical and
need their magnitude and later token effects inspected. If local predictions
agree yet output mismatches remain, investigate prefix commitment, rollback and
stopping logic. Sequential equality on one request is diagnostic, not a proof
for all requests.

The native DLite recovery now takes the greedy next token and pivot hidden from
the replayed prefix output, keeping them consistent with the restored cache.
Previously those values came from the discarded full-block output. This fixes a
state/output inconsistency but does not guarantee chunk/token bitwise equality.

## Text/image concurrency tables ignoring match gating

Run `EVAL_GPU=6 CONCURRENCIES="1 4 8" bash tests/compare_text_vlm_16.sh`.
This independently tests text, multimodal training prompts, and multimodal
validation prompts through the same HF reference
runtime, 16 requests per model/concurrency cell, 128 output-token cap, one warmup
by default. It includes AR, original DLite, local DLite 6/3 epochs, DFlash 3 epochs,
and DSpark 3 epochs, using the paths configured in `compare_vlm_16.sh`.
No group is gated by token matches or acceleration in another group.
The shell preflight checks that the chosen text prompts have no image and the
image prompts have one. Runtime/setup failures still abort.
Training and validation manifests must be different paths. The first 16 records
of each split are used by default, with the same sample/output caps for all models.
Training-set evaluation uses only image/question prompts: saved `answer` fields
are not supplied to the target and are not used as the generated output.

Default inputs and overrides:

| Group | Input | Override |
|---|---|---|
| Text | `$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main-0708-new/eval_datasets/gsm8k.jsonl` | `TEXT_DATA` |
| Multimodal train | `data/finevision_qwen35_sglang_20000/train.jsonl` | `TRAIN_EVAL_DATA` |
| Multimodal validation | `data/finevision_qwen35_sglang_20000/val.jsonl` | `VAL_EVAL_DATA` (legacy `EVAL_DATA` also supported) |

The image paths are relative to the workspace. Stage outputs are `text/`,
`multimodal_train/`, and `multimodal_val/`. Dataset split and absolute manifest
path are recorded in summaries and combined CSV/JSON. `--data-split` or shell
`EVAL_SPLIT` labels standalone comparisons. The split denotes the source manifest,
not proof that a specific checkpoint actually saw every selected record.

The current comparison wrappers support `MODELS` and `TESTS`:

```bash
MODELS="ar dlite_original" TESTS="text-only" EVAL_GPU=6 CONCURRENCIES="1 8" \
    bash tests/compare_text_vlm_16.sh

MODELS="ar dflash_3ep dspark_3ep dlite_3ep dlite_6ep" \
TESTS="visual-eval visual-training" EVAL_GPU=6 BATCH_SIZES="1 8" \
    bash tests/compare_static_batch_text_vlm.sh
```

Models: `ar dlite_original dflash_3ep dspark_3ep dlite_3ep dlite_6ep`.
Groups: `text-only visual-eval visual-training`. Defaults include all models and
groups. AR is always included as the speedup baseline; AR-only runs are allowed.
Selected order is preserved and only selected datasets/checkpoints are required.
The report writes only selected groups' tables. Typographical/duplicate selection
values fail rather than silently changing scope.

Text evaluation now directly reads single-string `turns` records from the user's
GSM8K JSONL without changing the prompt or rewriting the source file. Question
records remain supported. Multi-turn records are rejected; no user turns are
silently discarded. Dataset name (`TEXT_DATASET_NAME`, default `gsm8k`) and split
(`TEXT_SPLIT`, default `unspecified`) are recorded separately. The historical
text-first mismatch diagnostic still defaults to its original smoke prompts for
reproducibility; it is a separate script.

`speedup_tables.md` contains three tables: text, multimodal training set, and
multimodal validation set. Models are rows and concurrency levels are columns;
each cell shows speedup / accepted length, and each table names its data source.
These use **observed throughput ratio**: model output tok/s divided by AR output
tok/s at the same concurrency and in the same group.
This differs from the previous wall-time ratio when output token counts differ.
`comparison.csv` and `comparison.json` retain both ratios, tokens, wall time,
acceptance length, latency and token matches. This does not establish lossless
speedup; the usual correctness flags remain unchanged.

The opt-in CLI flag `--report-raw-speedup` (or `REPORT_RAW_SPEEDUP=1` for
`compare_vlm_16.sh`) displays observed throughput ratios and does not fail on token
mismatches. It adds `raw_wall_speedup_vs_ar` and
`raw_throughput_speedup_vs_ar`; `speedup_vs_ar` still follows the original strict
correctness gate. Without the flag, CLI mismatch failure behavior is unchanged.
Raw reporting requires block verification, never sequential diagnostics.

All three tables are single-GPU unbatched round-robin results. C=4/8 means several
independent request caches are active; target forwards are serialized rather
than continuously batched or overlapping. Do not interpret these tables as
SGLang/vLLM high-concurrency serving performance. Loading, warmup and CPU image
processing are excluded; GPU prefill and decode are included. Text and image
prompts differ, so compare each model with its within-modality AR baseline.
