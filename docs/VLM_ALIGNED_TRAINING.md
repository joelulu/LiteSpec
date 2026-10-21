# Image/text SFT aligned with the original DLite recipe

The default `scripts/dlite/run_training_vlm.sh` now runs
`litespec.vlm.train_aligned`. It inherits training hyperparameters directly from
`scripts/config/dlite_sft.yaml`, through
`scripts/config/vlm/sft_student_aligned.yaml`. The previous `vlm.train` implementation
and `sft_student_ddp.yaml` remain as legacy smoke-test code only.

## Effective defaults

| Setting | Aligned value |
| --- | --- |
| Draft depth / selected target layers | 5 / 12 |
| RNN sequential rank / block size | 320 / 8 |
| Maximum anchors | 512 valid anchors per sample |
| Sequence limit | 4096, including image tokens |
| Epochs | 6 complete epochs, no default update cap |
| Batch / accumulation | 1 image per rank / 4 microbatches |
| Eight-GPU global batch | 32 images per full optimizer window |
| Optimizer / precision | Upstream sharded Muon + AdamW / FP32 masters |
| Muon LR / momentum / decay | 0.02 / 0.95 / 0 |
| AdamW LR / decay | 0.0004 / 0 |
| Warmup / schedule | 0.04 / upstream cosine schedule |
| CE / probability L1 / base CE | 0.1 / 1 / 0 |
| Position decay / clipping | gamma 4 / global norm 1 |

The new adapter uses the original `OnlineDLiteModel` for anchors, local positions,
block isolation, teacher-forced sequential head and weighted greedy-target CE/L1;
`runner.run_draft` and `OptimizerWindow` for cross-epoch accumulation/global weight
normalization; and `BF16Optimizer`, parameter registration and FSDP layout adapter
for the original Muon partition, FP32 updates, clipping and scheduler. CUDA uses
FSDP1 `SHARD_GRAD_OP`, `use_orig_params=True`, with target embeddings/head ignored.
The original layout adapter requires **torch 2.9.1**. No environment upgrade is
needed if the existing VLM runtime still has that version.
The reused distributed helpers also import `yunchang`; the VLM environment now
declares `yunchang==0.6.4` and tqdm. Existing environments can add just these small
dependencies with `.venv-vlm/bin/python` as the explicit install target, without
re-resolving SGLang or the working Torch/Transformers stack.

HF Qwen3.5 target execution is the multimodal-specific change: the adapter passes
image tensors/grid/modality positions through the full frozen target, extracts
its language-layer hidden states and projects distinct needed text logits. The
draft continues using original local positions and no historical draft KV.
Only assistant-answer positions enter the supervision mask.

Rank zero validates the manifest once before training. To match the original
dataset filter, a record needs at least `2*block_size` answer labels and one full
block anchor. Short/overlength records are reported in `skipped_training.jsonl`.
Other exceptions (missing/corrupt images, invalid schema) abort validation rather
than silently changing the data. Eligible indices are broadcast, deterministically
shuffled and padded using the original DistributedSampler behavior. The resolved
config, eligible count and schedule are recorded in `effective_config.json`.
Operational logging defaults to local JSON (with optional explicit W&B), and save
cadence is shortened for this dataset. These do not change the training objective
or optimizer hyperparameters. Serving-only fields inherited from the text YAML
are not used by the HF image target adapter.

## Start a new six-epoch model

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
cd "$DLITE_WORKSPACE/LiteSpec"
git pull --ff-only origin research/qwen35-vlm
source .venv-vlm/bin/activate
unset MAX_STEPS MAX_SAMPLES CONFIG
export OUTPUT_DIR="$DLITE_WORKSPACE/runs/dlite_vlm_aligned_6ep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
TRAIN_DATA="$DLITE_WORKSPACE/data/finevision_qwen35_sglang_20000/train.jsonl" \
bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
```

Stop active generation clients/servers before using the GPUs. `MAX_STEPS` is now
**only an explicitly requested debug optimizer-update cap**. Full six-epoch
schedule length is retained in debug mode. Set `MAX_STEPS=2 MAX_SAMPLES=32` with a
separate output directory for the first GPU integration check, then unset both
for the real run. CPU numerical tests use an independent-block SDPA reference
because PyTorch FlexAttention has no CPU backward; CUDA never takes this fallback.

## Checkpoints and resume

The 2-layer/rank-64 smoke checkpoint is architecturally incompatible: start the
aligned 5-layer/rank-320 model from scratch. Checkpoints are immutable
`step_0000001` directories. They contain ordinary draft weights/config,
rank-local optimizer/FP32 master/scheduler/RNG/cursor state, and `vlm_run.json`.
A staging directory is atomically committed after all ranks finish; `last` is an
atomically updated symlink. Resume accepts only the same data/config/world size.

```bash
OUTPUT_DIR="/absolute/path/to/the/run" \
bash scripts/dlite/run_training_vlm.sh --resume /absolute/path/to/the/run/last
```

Scheduled saves occur every 4000 global microsteps at completed accumulation
boundaries, after the first update boundary reaching each epoch, and at completion.
The upstream runner carries partial windows across epochs: an epoch export may
therefore include the first 1-3 microbatches of the following epoch. No partial
gradient window is dropped or incorrectly resumed. Tail normalization follows the
original runner. Final total updates are `ceil(6 * ceil(N_eligible/world) / 4)`.

## Token-mismatch diagnostics

The observed real-checkpoint mismatch is not attributed to insufficient training.
The existing block verifier remains available and continues to fail visibly on
mismatches. `tests/run_vlm_eval.py --verification sequential` uses one-token target
calls on both AR and speculative paths, never inserting a rejected proposal into
the hybrid cache. This isolates block-vs-token numerical/cache behavior; it is
**not a throughput benchmark** and is not a replacement for efficient block
verification. Compare both modes on the same failing sample:

```bash
CUDA_VISIBLE_DEVICES=7 python tests/run_vlm_eval.py \
  --target-model "$HOME/LMM/lmm_model/Qwen3.5-4B" \
  --draft-model /path/to/checkpoint/last \
  --data ../data/finevision_qwen35_sglang_20000/val.jsonl \
  --sample-id 6c4d7cc250f10071cb0d24cc7f810bd9eeba74f0873017b5df711236eb946873 \
  --max-new-tokens 64 --verification sequential --output tests/results/mismatch_sequential.jsonl
```

The evaluator records first divergent token IDs and prints an aggregate acceptance
summary. Tiny real-image targets test sequential verification and rejection.
The supplied H100 checkpoint/image data is not accessible in the implementation
environment, so resolving that particular BF16 block mismatch and verifying the
8-H100 FSDP/Muon run still require host-side execution.
