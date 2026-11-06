# Pivot augmentation screen: separate training protocols (v2)

Branch: research/qwen35-pivot-augmentation. All models train from scratch,
2 epochs, up to 20k candidate rows, 8 GPUs. Defaults: microbatch 4,
accumulation 2, global batch 64, block 8, length 4096, image budget 1024.
Only final MODE/last inference weights/configs are saved. No step/epoch
checkpoints, optimizer/RNG dumps or training_state_rank files. YAML, validation
metadata, train.log and TensorBoard remain.

## Two independent comparison groups

| Protocol | Models | Control | Anchor budget | Target work |
|---|---|---|---|---|
| shared_sft_v2 | A B C E | A | up to 512/sample | one full Target pass on original microbatch |
| cached_rollout_v2 | D0 D1 D2 | D0 | 4 independent/sample | one cached prefix prefill + three short branches |

Do not attribute A vs D0 differences to feedback: the training protocols and
anchor budgets differ. Compare B/C/E with A and D1/D2 with D0. Each group has
identical common initialization, data, optimizer schedule and supervision budget.
The 512 SFT anchors are an upper limit; short answers naturally yield fewer.

* A: original 12 pivot layer slots.
* B: independent per-layer low-rank reads over four positions BEFORE the pivot
  (4 x 12 slots), depth/time identity, gated residual. Original attention remains.
* C: adjacent-position same-layer hidden difference through a shared bottleneck
  and gate, residual to the 12 pivot slots.
* D0: original pivot structure, cached TWO-round training, no feedback.
* D1: first rejected input position's 12 layer features, correct-minus-wrong
  boundary embedding and distance, gated residual to pivot.
* D2: same layer-wise feedback adapter; average its outputs into query index 1,
  which predicts the first candidate after the anchor. This is a one-position
  DFlow-style ablation, not complete DFlow.
* E ONLY: shallow (<7), middle (7..17), deep (>=18) banks with independent
  RMSNorm/Q/K/V; each query learns three weights. No stage grouping in B/C/D.

## Shared SFT

One frozen Target pass handles the original image/text microbatch. Many anchors
share its selected hidden and teacher distributions. Chunked/checkpointed Draft
forwards gather only each chunk's history/delta slots; no Target replication,
verification, second-round supervision or feedback is performed.
NUM_ANCHORS defaults to 512, SFT_CHUNK_BLOCKS to 16. The loss is the existing
DLite SFT CE/L1 objective, including its original supervision and decay rules.

## Cached rollout

The image encoder handles the ORIGINAL microbatch once. Small pooled visual
features are reused across independent anchor copies. Each copy has a different
prefix and therefore its own language cache; these prefixes are not shared
across different anchor positions.

Each anchor's correct prefix is prefilled ONCE, with left padding (never a
right-padded recurrent tail). Then:

1. Teacher-force B-1 correct tokens for first-round distributions; rollback.
2. Verify B anchor-inclusive actual candidate tokens; record rejection hidden;
   rollback to the exact original prefix.
3. Teacher-force at most 2B-1 corrected TEXT tokens from that same prefix;
   gather the second pivot and distributions at the actual acceptance boundary.

For B=8, branch widths are 7, 8, 15. Accepted short tokens may be replayed in
step 3; the image encoder and full prefix are NOT replayed. Rollback restores
attention length, convolution history and GDN recurrent matrices together.
Explicit MRoPE continuation coordinates account for left-padding length and
HF's unpadded rope delta. No variable-row recurrent commit is assumed.

Training remains two-round SFT: round-two future teacher-forced inputs retain
the source answer after the actual Target bonus; this is not full on-policy
answer generation. All Target paths, boundaries and feedback are detached.
EOS-ended round-two rows have zero weight. The same verification work runs for
D0/D1/D2. The old full_replay backend is an EXPLICIT numerical test reference,
never an automatic fallback. CPU regression compares cached and reference
hidden/pivot/logits/loss under image, text and mixed padded batches.

Additive training counters report prefix calls, short-branch calls, visual
encoder calls and processed token counts. These are summed over ranks/steps in
the logging window. Four Target calls can be cheaper than the old three full
sequence calls: only one call now processes the long prefix.

## Commands

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
export RUN_GROUP="$DLITE_WORKSPACE/runs/pivot_split_$(date +%Y%m%d_%H%M%S)"
TRAIN_SET=all bash tests/train_pivot_augmentation.sh
EVAL_GPU=7 EVAL_SAMPLES=16 TRAIN_SET=all bash tests/evaluate_pivot_augmentation.sh
```

TRAIN_SET=sft selects A/B/C/E; TRAIN_SET=rollout selects D0/D1/D2. EXPERIMENTS
can explicitly select a subset. Always choose a fresh RUN_GROUP; old final
models are not overwritten or used to warm start. Defaults remain 8 GPUs per
training run. MICRO_BATCH_SIZE and ACCUMULATION_STEPS affect both groups;
NUM_ANCHORS affects only SFT, ROLLOUT_ANCHORS only rollout.

Validation reuse is kept separately per group. SFT_REUSE_VALIDATION_FROM and
ROLLOUT_REUSE_VALIDATION_FROM can point to compatible prior validation metadata.
Legacy REUSE_VALIDATION_FROM is routed by its pivot_two_rounds flag; an old
rollout validation is never silently applied to SFT. Candidate/data/processor
hashes are checked. MAX_SAMPLES defaults to 20000 candidates, not guaranteed
eligible count. DEBUG_MAX_STEPS is the explicit debug update cap.

Evaluation is unchanged: fixed 16 samples/seed, B=1, greedy, per_token,
output cap 256, text + five image datasets + two video datasets. Tables record
training protocol, acceptance, first-candidate hit, decode raw ratio, prefill,
peak memory and matches. A is the SFT control; D0 is the rollout control.
Video-MME uses the existing description prompt, not official QA accuracy.
PIVOT_AUGMENTATION_OFF=1 disables added modules for a closed-loop ablation;
use a fresh RESULT_DIR. Different round boundaries are not paired causal probes.

CPU tests do not establish eight-GPU FSDP/CUDA throughput or memory. Keep the
working torch 2.9.1 environment; real speed improvements need host measurement.

## v3: shared supervision + trajectory-aligned feedback

New D0/D1/D2 runs use `joint_sft_rollout_v3`, not `cached_rollout_v2`.
A/B/C/E and all existing checkpoints retain their original behavior. From-scratch
initialization, final-only export, eight GPUs, 20k candidates and two epochs stay
unchanged. Never merge old/new result rows under the same training-protocol label.

All three D models now receive the SAME ordinary SFT supervision as A (up to
NUM_ANCHORS=512 per sample), PLUS ROLLOUT_ANCHORS=4 two-round starting points.
The original image microbatch is encoded once; detached pooled image features
are reused by full-sequence SFT and independent cached prefixes. Language prefix
prefills remain independent per rollout anchor. This is not a full-sequence hidden
cache reuse across different speculative prefixes.

The joint objective is `(SFT_mean + w * two_round_mean)/(1+w)`, with default
ROLLOUT_LOSS_WEIGHT=0.25. A detached rank-global denominator ratio balances the
branches so 512 SFT anchors cannot silently drown out four rollout anchors.
The existing runner still accumulates additive numerator/denominator terms;
across microbatches it weights the branch means by shared-SFT supervision mass.
Both first and second rollout rounds remain supervised. D0 is the no-feedback
control under this SAME joint objective; compare D1/D2 with D0.

TRANSITION_POLICY=trajectory (default) determines the accepted boundary by
matching actual proposals against the recorded Target answer and takes the bonus
from that answer. The teacher-forced continuation therefore stays on one recorded
trajectory. Feedback is still taken from an actual verifier pass on the proposed,
possibly wrong block. This follows DFlow's trajectory-guided transition idea,
NOT a full DFlow reproduction: we still reuse one rejected position, two total
rounds, and the existing DLite RNN head/loss. `verifier` explicitly restores the
old hybrid-trajectory intervention for comparison; there is no silent fallback.
Do not change inference verification to follow a dataset trajectory.

TensorBoard/logs add:
- `sft/loss`, `rollout/loss`, per-round loss and acceptance metrics.
- `supervision/sft_positions`, `supervision/rollout_round2_positions` (counts,
  not distinct answer tokens).
- `trajectory/anchor_root_disagreement`: cached Target root vs recorded anchor.
- `trajectory/verifier_token_disagreement`: actual candidate-conditioned verifier
  predictions vs recorded next tokens, INCLUDING speculative wrong-context effects.
- `trajectory/verifier_kept_disagreement`, `trajectory/verifier_bonus_disagreement`.
- `trajectory/round2_prev_greedy_disagreement`: RNN teacher-forced previous token
  vs Target greedy prediction at the preceding second-round position.
These are observations, not all implementation errors. In particular, a wrong
proposal can legitimately change later verifier predictions.

### Train the three v3 models

```bash
export RUN_GROUP="$DLITE_WORKSPACE/runs/pivot_feedback_v3_$(date +%Y%m%d_%H%M%S)_$$"
TRAIN_SET=rollout EXPERIMENTS="D0 D1 D2" \
NUM_ANCHORS=512 ROLLOUT_ANCHORS=4 ROLLOUT_LOSS_WEIGHT=0.25 \
TRANSITION_POLICY=trajectory EPOCHS=2 MICRO_BATCH_SIZE=4 ACCUMULATION_STEPS=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SAVE_TRAINING_STATE=0 \
bash tests/train_pivot_augmentation.sh
```

ROLLOUT_REUSE_VALIDATION_FROM may point at a compatible old D0/D1/D2 run; the
eligibility rule did not change. SFT validation remains separate. Validation
reuse never loads model weights. Old from-scratch A can be added explicitly as
an evaluation reference; do not train it again merely to run feedback probes.

### Paired feedback probe (works with OLD or NEW D checkpoints)

```bash
unset PIVOT_AUGMENTATION_OFF CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES
export RESULT_DIR="$RUN_GROUP/feedback_probe_$(date +%Y%m%d_%H%M%S)_$$"
EVAL_GPU=7 EXPERIMENTS="D0 D1 D2" TESTS="charxiv_val textvqa text-only" \
EVAL_SAMPLES=16 EVAL_MAX_NEW_TOKENS=128 PROBES_PER_SAMPLE=8 \
bash tests/diagnose_pivot_feedback.sh
cat "$RESULT_DIR/diagnostics.md"
```

No optimizer or parameter updates. Per checkpoint, construct a real draft/verifier
feedback record, then compare on/off/shuffled at the IDENTICAL second-round AR
prefix, pivot, anchor and reference. The prefix follows Target AR, never the
intervention's candidate. A deterministic cyclic DIFFERENT-SAMPLE donor changes
verifier hidden only; the correct/wrong boundary embeddings stay fixed. This
isolates hidden information from boundary identity. Different models can have
different eligible probe locations; pair conditions WITHIN each model, not their
raw probe means across models. Fixed-prefix accept is an AR-reference diagnostic,
not the candidate-conditioned closed-loop accept metric.

Reports: `diagnostics.md`, `summary.json`, `probes.jsonl`, input manifests,
AR sequences, checkpoint identities, image budget and settings. Logit changes,
proposal changes, first-token repair/break, CE, gate mean, actual model-dtype
residual ratio and changed-element fraction are recorded. Residuals are measured
AFTER BF16 rounding. D0 must have zero intervention effect and is a negative
control. Missing cross-sample donors/empty eligible probes fail explicitly.
These passes are UNTIMED and must not be used as speed benchmarks.

### Closed-loop test after training

Use a FRESH RESULT_DIR. The normal evaluator remains greedy B=1, per_token,
16 samples/seed42, image budget1024 and max-new-tokens256. For example:

```bash
export RESULT_DIR="$RUN_GROUP/eval_16_$(date +%Y%m%d_%H%M%S)_$$"
TRAIN_SET=rollout EXPERIMENTS="D0 D1 D2" EVAL_GPU=7 EVAL_SAMPLES=16 \
EVAL_MAX_NEW_TOKENS=256 TESTS="text-only charxiv_val textvqa" \
bash tests/evaluate_pivot_augmentation.sh
```

CPU tests establish tiny-model math/gradient/entrypoint consistency only, not
H100 memory headroom, fast-kernel equivalence, eight-GPU FSDP runtime or throughput.
