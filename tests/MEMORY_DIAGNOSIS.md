# Diagnose the trained vision reader without retraining

Branch: `research/qwen35-vlm-memory`. No changes to training, weights or normal
model execution. Uses the exact vision checkpoint recorded in the original
evaluation summary, rather than selecting an arbitrary newest `last` link.

```bash
RUN_GROUP="$DLITE_WORKSPACE/runs/dlite_memory_3ep_20261024_144948_3458849" \
EVAL_GPU=0 TESTS="charxiv_val textvqa" EVAL_SAMPLES=16 \
EVAL_MAX_NEW_TOKENS=256 PROBE_STRIDE=8 GRADIENT_SAMPLES=1 \
bash tests/diagnose_memory.sh
```

Requires the original `RUN_GROUP/evaluation` sample snapshots, per-case settings,
and checkpoint weights. Output defaults to a fresh
`RUN_GROUP/memory_diagnosis_TIMESTAMP_PID` directory. `RESULT_DIR` overrides it.
`PYTHON_BIN` can point to the working HF VLM environment from another worktree.
Only physical `EVAL_GPU` (default 0) is exposed; no training workers are launched.
The existing native HF environment is sufficient; no new dependencies are added.

## Check 1: same model, correct/wrong/disabled memory

Fresh isolated benchmark subprocesses evaluate AR plus the **same vision
checkpoint** in three conditions. Other trained baseline/text checkpoints are
not loaded. Each cell uses B=1, the first N samples of the saved seeded sample
snapshot, original image processor limits/attention backend/seed/state mode and
the same output cap. A fresh same-GPU AR baseline is reused for wrong/disabled.
Timing has no reader hooks or gradient diagnostics attached. Inspect the result
under each `*_correct`, `*_wrong_image`, `*_disabled` folder.

Wrong-image pairing comes from the benchmark's deterministic donor map, excludes
the same image path, preserves the original question, and changes only the
memory source. The primary target image, target cache and pivot stay correct.
Wrong-image end-to-end time contains donor prefill and is not a deployment rate.
Decode-only raw speedup is shown; token match is still recorded in per-cell JSON.

## Check 2: sensitivity and branch strength on a common prefix

Separately load the checkpoint and teacher-force a single AR token trace, one
native HF target token update at a time. Every eighth output position by default
is probed in all three conditions. The same pivot, anchor and target prefix are
used, and none of the Draft proposals is committed to this probe cache. Thus
this analysis does not confuse memory changes with divergent generation paths.
Its native single-token target execution is separate from the speculative GDN
backend used in the timed benchmark. All probe times are diagnostic, never rates.

Record each sample, generation position, source memory length, proposals,
gate mean/min/max and fraction below .01; intended residual/query norm ratio;
actual applied/query norm ratio after addition in model dtype; fraction of query
elements changed; logit absolute/RMS differences; probability L1/KL; proposal
token and complete-block agreement; correct/wrong residual cosine and difference.
The actual addition metric detects residuals that disappear under BF16 rounding.
The gate in this architecture depends only on pivot, so correct/wrong gate
equality by itself is expected, not a bug. Memory content should affect the readout.

`PROBE_ONLY=1` skips all fresh performance runs and reuses original AR traces
and donor assignments. This is quicker and produces the fixed-prefix analysis,
but not a new closed-loop speed comparison. It cannot extend an original trace
beyond its saved generation cap. `PROBE_STRIDE=1` probes all positions at higher
cost. `TESTS` can additionally select flickr30k/hr_bench_4k/hr_bench_8k; pure text
has no wrong-image condition and is deliberately excluded here.

## Gradient diagnosis and its limits

For one sample per dataset by default, backpropagate teacher-forced CE against
the next AR continuation tokens on the first probed prefix. Project raw memory
inside autograd so key/value/query/output/gate/norm gradients are all visible.
Report each reader parameter's norm, gradient presence/norm and finiteness.
Target is frozen. There is no optimizer, weight update or checkpoint save;
Draft gradients are cleared before and after. `GRADIENT_SAMPLES=0` disables it.

This is a **current held-out CE gradient probe**, not reconstruction of historical
training gradients, not a full replay of the original CE+L1 SFT objective, and not
proof that the module learned useful visual evidence during training. Old runs
without gradient logs cannot supply that history.

## Outputs and interpretation

- `diagnostics.md`: closed-loop table and fixed-prefix gate/sensitivity table.
- `paired_summary.csv/json`: aggregate metrics, with probe/sample counts.
- `*_paired.jsonl`: per-sample/per-position/per-condition evidence.
- `gradient_probe.json`: explicit probe objective and parameter gradients.
- `*_reader_parameters.json`: finite parameter checks and parameter norms.
- `settings.json`: checkpoint/config identity, selected sample IDs/hashes and args.
- `closed_loop_summary.json`, per-cell benchmark artifacts and logs.

Correct ≈ wrong ≈ disabled suggests little effective contribution. A nonzero
intended residual but zero applied delta indicates model-dtype rounding. Correct
and wrong readouts/logits that differ while acceptance stays equal means a branch
can affect representations without improving useful proposals. A small gate is
only one clue: evaluate residual magnitude and logits too. Do not equate any of
these observations with a proof that visual information is unnecessary.

CPU tests cover identical-logit controls, BF16 rounding, hook cleanup, zeroed
reader behavior, nonzero reader gradients without parameter updates, a real
hybrid target common-prefix walk, and the full CLI's three-condition/AR-reuse
pipeline. Actual checkpoint measurements still require the user's GPU server.
