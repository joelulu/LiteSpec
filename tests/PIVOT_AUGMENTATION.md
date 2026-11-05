# From-scratch pivot augmentation screen

Branch: research/qwen35-pivot-augmentation. A is the D0 common control.
All six modes use the same initial backbone seed, data, two actual draft/verify
rounds, optimizer schedule and loss. No warm start, no depth-stage grouping
outside E. Target is frozen; verification/transition/feedback are detached.

## Models

* A: original 12 pivot layer slots, no extra information.
* B: separate per-Draft-layer low-rank history attention over the four positions
  BEFORE the pivot (4 x 12 slots), layer identity plus relative time embeddings;
  gated residual, original attention denominator unchanged.
* C: per-source-layer adjacent-position hidden difference, shared bottleneck
  and gate, injected into the raw pivot. Still 12 conditioning slots.
* D1: first rejected input position's 12 layer features, conditioned on
  correct-minus-wrong token embeddings and relative distance, residual to pivot.
* D2: EXACT same layer-wise feedback adapter, average its 12 outputs and add
  only to query index 1 (DLite's first proposal hidden). The anchor query is
  index 0. This is a ONE-position DFlow-style ablation, not full DFlow.
* E: original attention remains. Each Draft layer additionally reads three
  banks using separate RMSNorm/Q/K/V. Source block ids <7 / 7..17 / >=18;
  each query softmax-weights three reads. Low-rank gated residual.

## Training and cost

Defaults: 20k candidate rows, 2 epochs, 8 ranks, microbatch 2, accumulation 4,
global batch 64; block 8, image budget 1024, length 4096, 5 Draft layers.
Eligibility requires 16 contiguous supervised tokens. Eligible count is logged;
20k candidate rows does not guarantee 20k eligible samples.

`ROLLOUT_ANCHORS=4` independently replicates each microbatch four times (each
replica has one sampled anchor). This is NOT the old 512-anchor SFT recipe.
All modes use this same budget. To fit smaller GPUs reduce ROLLOUT_ANCHORS
for ALL modes, never silently reduce just one mode. Full-prefix HF Target
passes occur three times: clean round 1, actual candidate verification, clean
corrected round 2. This prioritizes correct image/MRoPE/GDN handling over
training throughput; it does not add Target replay to inference.

Verifier tokens, not trajectory tokens, decide the boundary and bonus. Round 2
retains the original answer continuation AFTER that bonus as teacher-forced
inputs; labels/distributions are freshly computed by Target on the corrected
prefix. This is two-round SFT, not full on-policy answer rollout. Evaluable
second-round rows ending in EOS have zero loss. First-round loss remains.
Draft proposals suppress image/video placeholder token ids in all six modes,
so a random from-scratch proposal cannot create extra image placeholders during
full multimodal verification. This proposal-only rule is saved in the model
config and applied during inference too; Target predictions are not filtered.
The total objective is the combined valid-position weighted mean; detailed
round-specific CE/accuracy/acceptance metrics are recorded.

Only final `MODE/last/` weights/configs are exported; no step/epoch snapshots or
optimizer/RNG rank files. Validation metadata, one YAML, log and TensorBoard
remain. Validation reuse is checked against the two-round protocol.

## Commands

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
export RUN_GROUP="$DLITE_WORKSPACE/runs/pivot_aug_20k_2ep_$(date +%Y%m%d_%H%M%S)"
bash tests/train_pivot_augmentation.sh
EVAL_GPU=7 EVAL_SAMPLES=16 bash tests/evaluate_pivot_augmentation.sh
```

Evaluation: B=1, greedy, per_token, fixed samples/seed, eight datasets, output
256. Existing comparison tables show acceptance, decode throughput/raw ratio,
prefill, memory and token matches. Video-MME follows the existing description
prompt, not official QA accuracy. Different models have different round-2
prefixes; closed-loop comparisons are not paired prefix interventions.

Only static/video generation is supported for augmented checkpoints. Old
native/reference APIs fail explicitly. Set PIVOT_AUGMENTATION_OFF=1 during
evaluation to disable added paths on the same checkpoint (retain configuration).
Use a fresh RESULT_DIR for each ablation. Ablation paths are closed-loop unless
explicitly marked fixed-prefix; do not interpret different round boundaries as
paired causal interventions. DEBUG_MAX_STEPS=1 is an explicit eight-GPU smoke
cap; ordinary MAX_STEPS inherited from the shell is cleared by the launcher.
Source checks and CPU tiny-model tests run without GPUs; actual eight-GPU FSDP,
H100 kernels, memory usage and throughput must be validated on the training host.
