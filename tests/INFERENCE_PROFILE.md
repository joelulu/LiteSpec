# Qwen3.5 decode cost analysis

```bash
EVAL_GPU=6 TESTS="text-only charxiv_val" bash tests/profile_vlm_decode.sh
```

Defaults: AR + previous 100k-candidate visual DFlash checkpoint, B=1, 16 prompts,
128 new tokens. Override `MODELS` and normal checkpoint variables to include
original DLite, text DFlash or 20k visual DFlash. All existing TESTS selections
are supported. This does not change training or the current state recovery logic.

Three passes are deliberately kept separate:

1. Normal benchmark on all selected prompts, retaining the existing speedup
   definition. Results remain in comparison.csv / speedup_tables.md.
2. `PROFILE_SAMPLES=4` first prompts, full output cap, CUDA-event stage timing.
   Results go to each case's decode_profile.json, per-model *_profile.jsonl,
   and combined inference_profile.md. Each diagnostic sample is paired against
   its unprofiled output and reports instrumentation time / normal time.
3. `PROFILE_KERNELS=1`: one additional request per model/B, at most 32 new tokens,
   with torch.profiler CPU/CUDA trace and Python fallback-entry tracing. This
   request is not used for any speed figure. Set 0 to omit it.

Stages:

| Stage | Meaning |
|---|---|
| ar_target_decode | Normal one-token target forward and next-token selection |
| draft | Draft block forward, draft logits, sampling, candidate assembly |
| state_snapshot | Recurrent state snapshot and recording setup |
| target_verify | Whole candidate block target forward |
| acceptance | Target argmax, prefix matching and accept-length host readback |
| cache_restore | Rewind caches to pre-verify position and restore recurrent state |
| target_replay | Full target forward on the accepted prefix (current recovery) |
| cache_commit | Final cache crop / next pending token handling |
| hidden_features | Pivot extraction, and baseline history feature update |
| emit | EOS/control/output block handling |

CUDA timings are exclusive intervals along the current stream. They include
stream idle time between host launches; they are **not pure GPU kernel sums**.
Host timings are enqueue/control intervals, including waits at synchronizations;
they overlap CUDA work and must not be added to device intervals. Unassigned time
is reported rather than redistributed. No per-stage torch.cuda.synchronize is
inserted. CUDA events are resolved after the existing decode-end synchronization.
Profiling itself adds event/allocation/host overhead: use the separate normal
pass for speedups, and inspect Profile/normal time before interpreting percentages.

Kernel diagnostics distinguish package availability from actual execution:
`python_fallback_calls` counts entry into the HF Qwen3.5 Python fallback bodies,
tagged by sequence length, including prefill. A nonzero `torch_chunk...:seq=8`
confirms the fallback ran on that sequence size. Empty counts alone do not prove
all operations used optimal kernels; inspect cuda_kernel_names and Chrome traces.
Package presence or a wrapped Python function name is not execution evidence.
The trace probe is single-thread local, preserves existing Python profile hooks,
and never monkeypatches the model's GDN kernel functions.

For interpretation, inspect replay_fraction and target_calls_per_round first.
Compare target_replay, target_verify and draft ms/round. High replay cost motivates
state commit/local GDN replay; high hidden_features cost motivates removal of
redundant pivot work; confirmed fallback motivates kernel configuration checks.
Do not disable recovery just to obtain a speed number: this would corrupt state.

The decision is based on measured stages. No state-commit optimization, hidden
source change or Draft architecture change is included in this diagnostic patch.
