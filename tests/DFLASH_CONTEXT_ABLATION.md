# DFlash context and visual-slot ablations

Branch: `research/qwen35-dflash-context-ablation`, based on the tested pivot branch.
No rollout regeneration is necessary. Target inputs/weights, Draft block/head and objective stay unchanged.

## Training

Five from-scratch runs, same 20k validated FineVision rows, one epoch, block=8,
512 sampled supervised anchors, seed=42, 8 GPUs, micro=4, accumulation=2 (global=64).
Uses the original DFlash recipe (AdamW, LR 6e-4), not the DLite RNN or feedback recipe.
Target image budget=1024 tokens. One frozen Target full-sequence extraction per batch.
Only final `last` weights, configuration, preflight records, logs and TensorBoard are written.
Intermediate checkpoints and per-rank optimizer/RNG states are disabled. No warm start/resume.

| Context | Target positions available to each Draft block | Organization |
|---|---|---|
| full | all positions before anchor | original 5 selected layers, feature-concat and projection |
| recent128 | [anchor-128, anchor) | same 5-layer concat |
| pivot1 | anchor-1 | same 5-layer concat, one slot |
| pivot5 | anchor-1 | same 5 source layers as pivot1, separate depth slots and shared H-to-H projection |
| pivot12 | anchor-1 | 12 separate slots with learned depth embeddings and shared H-to-H projection |

12 layer IDs are `[0,1,4,7,11,14,17,20,24,27,30,31]` (language block outputs, final norm excluded).
All pivot5/pivot12 depth slots carry the same absolute time position; depth embedding distinguishes layers.
The bidirectional intra-block query attention is unchanged. This is not the architecture's causal SWA mode.
Padding/inactive blocks cannot expose other blocks or future Target positions.
Training masks source positions; inference physically retains <=128/1 raw positions for bounded modes,
rebuilds their small K/V each round, and never retains speculative query K/V.
Full uses the original projected-history cache. All variants' measured Draft time includes their actual projection/cache work.
Pivot12 changes feature sources, organization and projector parameter count; it is not a pure slot-count ablation.
The bounded-context native text `spec_generate` entry fails explicitly; use the supplied static VLM entry.

From a fresh terminal (set WORKSPACE first):

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
cd "$DLITE_WORKSPACE/LiteSpec"
git fetch origin research/qwen35-dflash-context-ablation
# Use a fresh directory; avoid overwriting an existing worktree.
git worktree add --detach "$DLITE_WORKSPACE/LiteSpec-dflash-context" FETCH_HEAD
cd "$DLITE_WORKSPACE/LiteSpec-dflash-context"
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
export RUN_GROUP="$DLITE_WORKSPACE/runs/dflash_context_20k_1ep_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
  bash tests/train_dflash_context.sh
```

Use `bash tests/run_dflash_context_ablation.sh` instead to train and then run all evaluation automatically.
It clears a stale RESULT_DIR and assigns a fresh run group unless you explicitly set RUN_GROUP.

Optional `REUSE_VALIDATION_FROM` must point at a compatible existing preflight. Otherwise full validates once,
and the other variants reuse its exact eligible set. `CONTEXTS` selects subsets; one run group must be fresh.
`TRAIN_SAMPLES`, `EPOCHS`, `MICRO_BATCH_SIZE`, `ACCUMULATION_STEPS` are explicit overrides.
The script unsets stale MAX_STEPS and initial/resume paths. `MAX_LENGTH` is set in the YAML (4096).

## Evaluation

```bash
# Keep/export the actual completed RUN_GROUP from above.
EVAL_GPU=7 EVAL_SAMPLES=64 EVAL_MAX_NEW_TOKENS=256 \
  TESTS="charxiv_val textvqa" bash tests/evaluate_dflash_context.sh
```

Defaults: B=1, seed42, image1024, `per_token` Target state management. No training occurs here.
`DFLASH_100K` optionally overrides the historical checkpoint (actual eligible sample count must be checked before publication).
Default is `runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dflash/last`.
Evaluation requires the original four checkpoints plus that old full-context checkpoint. Pivot5 is automatically added when `RUN_GROUP/pivot5/last/config.json` exists or `PIVOT5_CHECKPOINT` is supplied. `EVAL_CONTEXTS` explicitly selects context variants (include full for the visual diagnosis).

1. Correct-context closed-loop: AR + context checkpoints + old DFlash.
2. Wrong/Mask closed-loop: new Full + old DFlash, reusing same-GPU/input AR reference.
3. Fixed-AR-prefix probes: both Full checkpoints; equal prefixes, anchor/pivot and references
   within each checkpoint. Candidate proposals never enter the AR Target cache.

Wrong intervention swaps only visual-position Target hidden (before Draft feature projection and RoPE),
using the next distinct processed image in the fixed sample order. Donor features include that donor's original
prompt context; this is a visual-interface intervention, not an isolated image-encoder intervention.
Uniform rounded indices align unequal visual lengths while preserving recipient slot count/positions.
Mask blocks attention to those same slots; it does not just zero Value vectors.
Visual slots include vision delimiters as well as image/video placeholders, matching the existing visual mask.
Target always receives the correct image and retains all its context. Text hidden and pivot still contain visual information.

Wrong donor prefill is outside reported normal timing; its speed/latency are diagnostic, not deployable speedup.
Closed-loop prefixes can diverge. Fixed-prefix reports are the causal comparison:
accept (including guaranteed anchor), first candidate hit (excluding anchor), repair/break vs Mask,
and candidate change vs Correct. Samples with no complete proposed block cannot contribute a fixed-prefix probe.
Probe counts, donor IDs/lengths, absolute anchors, proposals/references and AR trajectories are saved.
The old/new training budgets differ: interpret interventions within checkpoint, not as a controlled training-scaling curve.

Results: `comparison.md/json`, per-cell `summary.csv/json` and per-request JSONL;
`visual_probe/diagnostics.md`, `summary.json`, `probes.jsonl`, `ar_sequences.jsonl`, settings and selections.
Raw ratios do not establish lossless decoding or task accuracy; AR token matches remain visible.

## Validation

CPU tiny Qwen3.5 integration covers training/backprop, export/reload, bounded/full cached decode,
`per_token` GDN commit, Wrong/Mask, and the actual fixed-prefix CLI. Mask tests compare dense/Flex rules
and compare packed training hidden against independent same-prefix inference.
Full-size H100 eight-rank training and CUDA Flex kernels must still be exercised on the user's machine.

## PyTorch 2.9.1 CUDA callback compatibility

The first release used `&=` in a Flex mask callback, which lowered to `aten.copy_` and
failed Inductor pointwise-subgraph lowering on CUDA. All DFlash mask combinations now
use out-of-place boolean expressions. Visibility and the training objective are unchanged.
Regression checks prohibit augmented assignment inside the callback and inspect its functionalized FX graph.
Before allocating eight ranks, run:

```bash
CUDA_VISIBLE_DEVICES=7 "$PYTHON_BIN" tests/check_dflash_context_cuda.py
```

This compiles and backpropagates actual recent128/pivot1/pivot5/pivot12 Flex kernels with head_dim=256
and compares outputs to dense SDPA. It requires a CUDA machine; it was not executed in the CPU editor environment.
A completed Full checkpoint remains valid. Restart the three bounded variants in a fresh run group
using `CONTEXTS="recent128 pivot1 pivot12"` and `REUSE_VALIDATION_FROM=<old-run>/full`.
After training, symlink `<new-run>/full` to `<old-run>/full` before the combined evaluation.

## Supplement Pivot5 without retraining existing checkpoints

Pivot5 uses `[1,8,15,22,29]` for Qwen3.5-4B, matching Pivot1. Each source
layer contributes one slot, with learned depth embedding and shared H-to-H projector,
exactly as in Pivot12's organization. All slots use the same pivot time position.
Block size remains 8; prediction head/objective and 20k/1-epoch budget are unchanged.
Pivot1 vs Pivot5 also changes projector parameter count; report it with the organization change.

```bash
export COMPARISON_RUN="$DLITE_WORKSPACE/runs/dflash_context_fixed_20261110_062258_1299222"
export PIVOT5_RUN="$DLITE_WORKSPACE/runs/dflash_pivot5_20k_1ep_$(date +%Y%m%d_%H%M%S)_$$"
RUN_GROUP="$PIVOT5_RUN" CONTEXTS=pivot5 \
REUSE_VALIDATION_FROM="$DLITE_WORKSPACE/runs/dflash_context_20k_1ep_20261110_041837_1299222/full" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
  bash tests/train_dflash_context.sh

unset RESULT_DIR
RUN_GROUP="$COMPARISON_RUN" PIVOT5_CHECKPOINT="$PIVOT5_RUN/pivot5/last" \
  EVAL_GPU=7 EVAL_SAMPLES=64 bash tests/evaluate_dflash_context.sh
```

No copying or overwriting of the four completed checkpoints is needed.

## Video visual-slot probes (no training)

`bash tests/probe_dflash_visual_video.sh` probes the original Full-2w and historical
DFlash-100k checkpoints on TextVQA, Video-MME and VideoDetailCaption, 50 distinct inputs
per dataset, seed42, cap256. Up to8 fixed AR-prefix probes/input, stride16. Thus N counts
probe prefixes, not source samples; short/EOS outputs can yield fewer than400 prefixes.
Video defaults:16 evenly sampled frames,200704 pixels/frame; TextVQA image budget1024.
`FULL_2W`, `DFLASH_100K`, `VIDEO_DATA_ROOT`, `TESTS`, `EVAL_GPU`, and `RESULT_DIR` are overridable.
Use a fresh RESULT_DIR. No training or performance benchmark is run by this entry point.

Video uses the existing native video preparation, timestamps and prompts. Video-MME
continues to use the repository's description prompt, not official multiple-choice accuracy.
Wrong swaps another video's visual-position Target features from within the same dataset.
Recipient text, pivot, positions and visual-slot count remain fixed; unequal donor lengths
use the same deterministic index alignment as images. Both image and video probe masks
include visual delimiters. Target always retains the correct original image/video.
Processed grids/frame counts and intervention-slot counts are recorded in settings.json.
