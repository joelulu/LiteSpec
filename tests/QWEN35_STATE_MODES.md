# Qwen3.5 HF state recovery experiment

No retraining or checkpoint conversion. The existing HF static-batch evaluator
now accepts `--state-mode` (shell: `STATE_MODE`). Default behavior is unchanged.

| Mode | Verify | Partial-accept commit |
|---|---|---|
| `full_target_replay` | Existing native HF chunk/recurrent kernel | Restore start state, crop entire proposal, run full Target on accepted prefix |
| `per_token` | Fused recurrent kernel, save FP32 state after each token | Copy accepted-position state; crop conv history and attention KV |
| `compact_replay` | Same fused recurrent kernel, retain start state and post-conv k/v/g/beta | One fused **state-only** replay per GDN layer; crop conv history and attention KV |

The two new modes retain the original verification hidden/logits at the accepted
position. The old path retains its existing replay-hidden/replay-logit behavior.
This can introduce numerical differences; cross-mode token and acceptance
equality are reported rather than assumed. A test with exact original HF hidden
replay semantics would require the extra Target forward we intend to eliminate.

## Source audit and scope

Inspected on 2026-10-23:

- vLLM `f4edf9f0f078ae67392510aafd1871288948a3c3`,
  [`fused_gdn_decode_post_conv_mtp.py`](https://github.com/vllm-project/vllm/blob/f4edf9f0f078ae67392510aafd1871288948a3c3/vllm/third_party/flash_linear_attention/ops/fused_gdn_decode_post_conv_mtp.py):
  recurrent verify starts from the previous accepted state and stores each
  candidate position's state. Its state layout is `[slots, HV, V, K]`.
- SGLang `8b9f72417fa85a04de6193b0a52f4cec63370d77`,
  [`gdn_backend.py`](https://github.com/sgl-project/sglang/blob/8b9f72417fa85a04de6193b0a52f4cec63370d77/python/sglang/srt/layers/attention/linear/gdn_backend.py)
  and [`gdn_replayssm_spec_decode.py`](https://github.com/sgl-project/sglang/blob/8b9f72417fa85a04de6193b0a52f4cec63370d77/python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py):
  intermediate-state selection and optional circular compact-record recovery.
  The latter uses checkpoint + d/k/g history, output-only reconstruction, and
  periodic folding; exact-fold variants retain additional precision/raw records.
- ReplaySSM reference fork `73a9592e9d8a2268620e8883a82f5cec7442ed36`,
  [`gdn_replayssm_spec_decode.py`](https://github.com/Johnny-Liou/ReplaySSM/blob/73a9592e9d8a2268620e8883a82f5cec7442ed36/vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py)
  and its speculative kernel tests: circular cursor commit, output-only verify,
  periodic checkpoint flush. Its tests explicitly allow arithmetic-order
  differences between chunk UT-transform and recurrent kernels.
- Installed Transformers 5.20.1 Qwen3.5: native cache layout `[B,H,K,V]`;
  cache crop restores convolution history but not recurrent state. Our adapter
  follows its projection, gating, normalization, and convolution order.

`compact_replay` implements the **per-block checkpoint + compact update inputs**
idea. It is NOT a port of the full upstream ReplaySSM circular/output-only
algorithm and must not be labeled an upstream ReplaySSM performance result.
It materializes the final state every block and replays only after rejection.
Moving to across-round circular history is a separate extension.

The existing replay path uses the native HF verifier; both new modes share a
custom Triton recurrent verifier. Therefore old-vs-new includes a kernel change;
per-token-vs-compact isolates the recording/commit strategy. All methods keep
the same Target and Draft weights, image budgets, prompts, and decode limits.

## Run

From `LiteSpec/`, with the existing `.venv-vlm` environment:

```bash
export RESULT_DIR="$DLITE_WORKSPACE/runs/qwen35_state_modes_$(date +%Y%m%d_%H%M%S)"
EVAL_GPU=6 \
MODELS="ar dlite_original dflash_vlm" \
TESTS="text-only charxiv_val" \
BATCH_SIZES=1 EVAL_SAMPLES=16 EVAL_MAX_NEW_TOKENS=128 \
PROFILE_SAMPLES=16 \
bash tests/compare_qwen35_state_modes.sh
cat "$RESULT_DIR/state_modes.md"
```

`DFLASH_VLM` defaults to the existing 100k-candidate/3-epoch/block-8 checkpoint.
Override it to evaluate another trained Draft. All existing MODELS/TESTS/image
budget selectors are supported. For five visual datasets, use
`TESTS="charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k"`.

The script first runs a CUDA numerical check with FP32/BF16 inputs, nonzero
initial state, and every accepted prefix. It fails before benchmarking if this
check fails. No package/environment upgrades are performed. CUDA requires the
existing Triton and causal-conv1d installation; no silent Python fallback.

## Measurements

- `state_modes.md` / `.csv`: normal decode tok/s, vs-AR and vs-full-replay ratios,
  acceptance, peak CUDA allocation, retained transaction-record bytes, parity.
- `<mode>/inference_profile.md`: separately instrumented stage intervals.
- `<mode>/<case>/decode_profile.json`: calls/round, full Target replay, state
  selection/replay, cache crop, and optional actual CUDA kernel names.
- `<mode>/<case>/*_b1.jsonl`: per-request tokens and acceptance lengths.
- `cuda_state_check.json`: GPU state-kernel validation results.

Normal inference is timed separately from profiling. All custom tail-width
kernel variants are compiled before timing; accepted-prefix length is a runtime
argument, avoiding a new compilation for each acceptance length. Kernel probes
are a third, separate short pass. CUDA stream intervals include host launch
gaps; host times overlap device times and must not be added.

`peak_transaction_record_mib` counts retained GDN transaction payload only
(baseline snapshots, per-token matrices, or checkpoint + update inputs). It
excludes convolution history, attention KV, temporary outputs, and allocator
overhead. `peak_allocated_gib` includes those allocations and model weights and
may be dominated by prefill.

Supported scope: eager native HF Qwen3.5, unpadded replicated static batches,
linear candidate chains, FP32 recurrent cache. No serving scheduler, tree
verification, mixed-length batch, torch.compile, or CUDA-graph support claim.
CPU uses a slow reference for correctness tests only.

## Local checks

```bash
PYTHONPATH=. python -m pytest tests/test_gdn_state.py tests/test_vlm_static_batch.py tests/test_inference_profile.py -q
```

The tests exercise every prefix of a short block against a direct Target
forward, recurrent/conv/KV recovery, text/image multi-round decoding, B=1/2,
full acceptance, EOS, truncated blocks, adapter cleanup, CLI/report integration,
and optional CUDA recurrence/state selection tests. GPU performance must be
measured on the server; CPU success and offline SM90 compilation do not establish
H100 numerical correctness or speed.
