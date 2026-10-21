# Eight-GPU generation and image/text baselines

## One command owns the serving lifecycle

Copy `scripts/data/generate_finevision_8gpu.sh` into the parent `dlite-vlm/`
directory if desired. It works both there and at its original repository path.
The client uses `.venv-vlm`; eight independent TP=1 SGLang replicas use the
separate `dflash` environment. This is eight internal endpoints for throughput,
not an eight-way tensor-parallel model. A single command manages them all.

```bash
SGLANG_PYTHON="$HOME/.conda/envs/dflash/bin/python" \
SOURCE_DATA="$HOME/LMM/liyt/datasets/FineVision/FineVision_subset/finevision_sharegpt.jsonl" \
TARGET_MODEL="$HOME/LMM/lmm_model/Qwen3.5-4B" \
DATA_GPUS=0,1,2,3,4,5,6,7 \
MAX_SAMPLES=20000 MAX_NEW_TOKENS=256 MAX_LENGTH=4096 \
IMAGE_MAX_TOKENS=1024 CONCURRENCY=8 \
PREPARE_OUTPUT="/absolute/path/to/new/data/directory" \
bash generate_finevision_8gpu.sh
```

| Parameter | Default / meaning |
| --- | --- |
| MAX_SAMPLES | 20000 unique eligible single-image first-turn prompts; 0 means all |
| MAX_NEW_TOKENS | 256 maximum output tokens; thinking disabled, temperature 0 |
| MIN_ANSWER_TOKENS | 1 supervised token including chat end marker; explicitly set 16 for the old block-8 collection filter |
| ADAPTIVE_ANSWER_BUDGET | 1: shorten output cap to remaining context minus 8 suffix tokens; 0: require full cap + 32 |
| MAX_LENGTH | 4096 maximum total processor sequence length |
| IMAGE_MAX_TOKENS | 1024 merged visual-token cap; 0 retains original images |
| CONCURRENCY | 8 requests per replica, 64 total |
| BASE_PORT | 30000, using consecutive ports through 30007 |
| STARTUP_TIMEOUT | 1800 seconds to wait for all eight services |
| CONTEXT_LENGTH | Defaults to MAX_LENGTH; cannot be smaller |
| MEM_FRACTION_STATIC | 0.80 per serving replica |
| PREPARE_OUTPUT | `../data/finevision_qwen35_sglang_<N>_img<T>` |

FineVision extraction keeps the previously confirmed `conversations[0].content`
and first-turn-only mapping. Deduplication occurs before counting the sample cap.
The cap counts prompts, not guaranteed successful outputs: oversized prompts and
failed requests are reported separately. Failed HTTP/answer records are retried
on rerun; permanently skipped prompt records are tracked. Short answers are kept
by default during collection. This does **not** change upstream training's
`>= 2 * block_size` label/full-anchor eligibility: a saved short answer may still
be skipped by training. Train/val partitioning also means a 100,000-candidate run
does not imply 100,000 training examples. Use a fresh output directory when
changing filtering policy; resume does not revisit old skipped rows.
Identical settings are
required for resume. New manifest schema rejects older manifests whose limit
semantics counted candidates before deduplication; use a new output directory.

Image bounds derive from the local processor's patch and merge sizes and native
min/max pixels, never a hard-coded 28-pixel factor. For patch=16, merge=2, the
1024-token ceiling corresponds to 1,048,576 pixels (approximately 1024-square).
Aspect ratio is retained approximately, with dimensions aligned to the merge
factor. The processor's native minimum still applies. Some extreme aspect ratios
cannot meet narrow pixel bounds and are reported as skipped. EXIF orientation is
applied before resizing; original images remain untouched. Resized PNGs are
written to `images/`, sent to the service, validated on CPU, and referenced by
absolute path in training records. Keep this directory with the generated data.
Do not move it without rewriting the recorded paths. New bounded-image datasets
split by source image content hash so byte-identical files cannot cross splits.
Without resizing, the legacy path-based split remains unchanged.

The client compares server-reported prompt-token counts to its own processor
encoding when usage is available. This catches many tokenizer/processor/template
differences, but does not prove identical floating-point image representations.
Generation metadata and image policy are persisted. No image-specific processing
kwargs are assumed to be accepted by SGLang.

The manager checks all ports are free, checks eight visible GPUs in the SGLang
environment, starts isolated process groups, waits for `/health` and the expected
`/v1/models` identity from every replica, then starts regeneration. It monitors
server liveness while generating. Success, exceptions and signals all clean up
the owned client/server process groups. It never adopts or kills pre-existing
services. Logs and PIDs are in `serving_logs/`. API clients bypass proxy settings
for loopback URLs. No SGLang packages are installed into the training environment.

## DFlash and DSpark controls

```bash
ALGORITHM=dflash REPORT_TO=tensorboard \
TRAIN_DATA=/absolute/path/train.jsonl EVAL_DATA=/absolute/path/val.jsonl \
bash train_eval_vlm.sh all

ALGORITHM=dspark REPORT_TO=tensorboard \
TRAIN_DATA=/absolute/path/train.jsonl EVAL_DATA=/absolute/path/val.jsonl \
bash train_eval_vlm.sh all
```

The common launcher selects `sft_student_aligned.yaml`, `sft_dflash.yaml`, or
`sft_dspark.yaml`. The evaluator resolves the checkpoint architecture and uses the
same image processor, frozen Qwen3.5 target and hybrid-cache verification.

| Setting | DLite | DFlash | DSpark |
| --- | --- | --- | --- |
| History | Pivot only, no draft history KV | Full target-history draft KV | Full target-history draft KV |
| Layers / block / epochs | 5 / 8 / 6 | 5 / 8 / 6 | 5 / 8 / 6 |
| Selected target layers | 12 | 5 | 5 |
| Sequential head | RNN rank 320 | None | Vanilla Markov rank 256 |
| Original objective | 0.1 CE + 1 L1 | Token CE | 0.1 CE + 0.9 L1 + confidence loss |
| Optimizer | Muon + AdamW | AdamW | AdamW |
| Learning rate | Muon .02 / AdamW .0004 | .0006 | .0006 |
| Per-GPU batch / accumulation | 1 / 4 | 1 / 4 | 1 / 4 |

Baseline model and loss implementations are reused unchanged from LiteSpec.
Overrides of the original baseline configs are explicit: DFlash's block 16 is
changed to 8; both baselines' accumulation 1 becomes 4; positional loss decay is
4. This is a common-block/common-batch control with method-specific optimizers,
not a claim of identical recipes or a reproduction of published headline results.
All adapters use full frozen HF image inputs, answer-only masks, shared short
answer/full-anchor filtering and the original runner/FSDP optimizer plumbing.
DFlash/DSpark preserve their original hard labels and losses; DLite preserves
its greedy-target CE labels. Keep the same dataset and evaluation IDs in all runs.

Baseline decoding projects new committed target features into persistent draft
KV and crops speculative-query KV before the next round. It uses the original
proposal hooks (parallel DFlash and Markov DSpark) with a fixed block; adaptive
DSpark confidence scheduling is not enabled. Draft positions use the original
one-dimensional DFlash convention; target visual positions remain Qwen3.5 MRoPE.
Only full-history baseline inference is supported in this first adapter.

These are batch-one HF reference evaluations. Sequential verification is an
acceptance/correctness diagnostic; block verification includes full hybrid-cache
snapshot/replay costs. Neither is a production SGLang throughput benchmark.
Never compare a previously evaluated checkpoint with a new one while silently
changing verification modes, image budgets or prompt data.

## Audit and verification scope

The rejection path previously restored target cache but retained hidden/logit
outputs from the unrecovered speculative chunk. It now uses replayed outputs as
the authoritative input to the next round. This removes an internal state/output
inconsistency; it does not claim to eliminate BF16 chunk-vs-token differences.
Mismatch reporting remains enabled.

CPU tests exercise real tiny Qwen3.5 image forwards, frozen-target gradients,
baseline optimizer updates, saved checkpoint reload, cached multi-round generation
and AR token equality under both verifiers. Other tests cover image materialization,
dedup/resume identity, HTTP concurrency and errors, owned-process cleanup and
server failure detection. Full eight-H100 SGLang/FSDP execution still requires the
user's serving/training environments and real dataset. A passing CPU suite does
not establish large-model numerical equality or production throughput.
