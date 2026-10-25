# Diagnose the two trained GDN checkpoints

Run in the GDN worktree with the existing environment, in the foreground:

```bash
bash tests/diagnose_gdn.sh /absolute/path/to/completed/run_group
```

With exactly one completed `runs/vlm_gdn_valid20k_*` group, the path can be
omitted. Otherwise the script lists candidates and requires an explicit path.
It reads `dflash_gdn/last`, `dlite_gdn/last`, and `evaluation/inputs/*.jsonl`.
The Target and image cap come from the exported `vlm_run.json`; checkpoints
must agree. The original models and training files are never overwritten.

Defaults: physical GPU 7, eight saved evaluation samples per dataset, native
Target-only greedy continuation capped at 64 tokens, anchors at generated-token
offsets 0,8,16,32. Short references use fewer anchors/positions and are listed in
`skipped.json`. This is a small diagnostic, not another speed benchmark.

```bash
DIAG_GPU=7 DIAG_SAMPLES=16 bash tests/diagnose_gdn.sh /absolute/path/to/run_group
```

## 1. Saved weights, load consistency and gradients

`checkpoint_audit.json` includes:

- Every Reader parameter's shape, dtype, norm, finite/nonzero counts.
- Full checkpoint/model missing and unexpected keys (including sharded
  safetensors), plus exact saved-versus-loaded Reader equality after dtype cast.
- Whether `up.weight`, whose original initialization was zero, is now nonzero.
- Checkpoint cursor/config and any global gradient norms found in `train.log`.

Missing/ignored/nonfinite Reader weights fail before the mode comparisons.
Nonzero `up.weight` establishes a change from its zero initialization. It does
not, by itself, prove the Reader learned a useful conditional prediction rule.

**The original training did not log Reader-specific historical gradients.**
Existing global gradient norms do not establish those gradients. The report
explicitly records that historical Reader gradients are unavailable.

`gradient_probe.json` computes a NEW diagnostic teacher-greedy CE backward probe
on the first usable validation anchor for each checkpoint, reporting every
Reader parameter's gradient connectivity, finite values, norm and nonzero count.
It uses `torch.autograd.grad`, performs no optimizer step, and does not mutate
parameters or their `.grad`. This tests current gradient flow, not historical
training updates or exact reproduction of the original weighted CE/L1 objective.

## 2. Residual, logits and proposals

`anchors.jsonl` records `norm(delta_h)/norm(h)` at the existing injection point,
restricted to prediction positions. For normal/off/wrong modes it also records:

- Teacher-forced logits mean/max absolute change and KL versus normal mode.
- Teacher-forced top-1 change fraction.
- Actual free-running draft candidate token IDs and change fraction. DLite uses
  its recurrent sequential sampler here; teacher-forced argmax is NOT substituted
  for its real inference proposals.
- CE versus target greedy labels, CE versus the fixed reference continuation,
  and L1 distance to target probabilities.

Zero residual and identical logits indicate no observed forward influence on
that anchor. Nonzero residual/logit changes with no candidate change mean the
influence did not change the sampled greedy proposals there.

## 3. Same checkpoint: normal / off / wrong sample

Each input's Target-only reference token IDs are generated once, saved in
`references.jsonl`, and shared by both checkpoints and all modes. At each anchor,
the Target receives only that committed reference prefix, with native image
inputs and MRoPE. Its GDN states are copied BEFORE executing pending/future tokens.
All modes share the recipient pivot, target-history features and reference
logits. Only the Reader's state or its entire residual branch changes:

| Mode | State / intervention |
| --- | --- |
| normal | Recipient's causally aligned committed state |
| off | Reader is bypassed as an identity, preserving the SAME checkpoint backbone |
| wrong | State from a different image; text-only cases use another question |

Wrong donors are other saved inputs, never replicated copies of the same input.
They use the same generated offset when available, otherwise a clamped offset.
Both offsets and prompt lengths are reported. Their different context lengths
are a limitation: a wrong-state penalty alone does not isolate visual semantics.
The intervention replaces joint visual/text GDN state, not pure visual state.

`fixed_reference_committed_length` is 1 plus the consecutive matching candidate
prefix on the fixed native-AR reference, capped by available reference tokens.
It is a controlled drafting-quality diagnostic, not the measured speculative
acceptance of the earlier `per_token` verifier and not a proof of AR equivalence.

Outputs under a NEW `run_group/gdn_diagnostics_<time>_<pid>/` directory:

- `checkpoint_audit.json`: saved/load/update checks and historical evidence limits.
- `gradient_probe.json`: new per-parameter diagnostic gradients.
- `anchors.jsonl`: all per-anchor interventions and changes.
- `summary.csv`, `summary.json`, `report.md`: dataset/model/mode aggregates.
- `references.jsonl`, `settings.json`, `skipped.json`: fixed inputs, settings and coverage.

CE/L1/KL/flip rates are weighted by supervised positions. Residual ratios and
reference committed lengths are equal-anchor means. No latency conclusions
should be drawn from this instrumented diagnostic.

## How to interpret

- Saved Reader absent or inconsistent after loading: fix serialization/loading.
- Output projection still zero: investigate original optimization and gradient
  handling; a fresh nonzero gradient does not prove training previously used it.
- Parameters changed but residual/logit influence tiny: inspect scale and fusion.
- Normal and wrong behave alike across sufficient anchors: no evidence that
  predictions depend usefully on the current sample's state.
- Normal beats off and wrong: state contributes useful conditional information.
- Wrong hurts but off beats normal: conditional information is used, but its net
  integration is harmful on these anchors.

Tiny-model CPU tests cover load mismatch/missing keys, zero initialization,
Reader bypass restoration, production candidate parity for both families,
nonzero residuals/state sensitivity, gradient probes without weight updates,
and the complete report CLI with real native-image Target execution. Actual
trained 4B checkpoint results must be collected on the host.
