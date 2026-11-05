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
