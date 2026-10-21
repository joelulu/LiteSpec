# Sixteen-prompt checkpoint comparison

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

Each run writes `settings.json` (checkpoint paths and exact sample IDs),
`summary.csv`, `summary.json`, and one JSONL per model/concurrency with output
tokens, acceptance lengths, timing and match flags. The terminal prints a compact
summary. Any mismatch makes the final command fail and invalidates the affected
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
