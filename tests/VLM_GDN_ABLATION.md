# Qwen3.5 VLM GDN reader experiment

Branch: `research/qwen35-vlm-gdn`, independently based on `b72163b`.
No memory-branch commits, pivot memory modules, or checkpoints are incorporated.
Target: frozen native HF Qwen3.5-4B. Train and evaluate four NEW checkpoints:

| Arm | Draft history KV | Additional target GDN state |
| --- | --- | --- |
| `dflash` | Full existing DFlash cache | None |
| `dlite` | None; existing pivot CHS | None |
| `dflash_gdn` | Full existing DFlash cache | ReaDy-style residual reader |
| `dlite_gdn` | None; existing pivot CHS | ReaDy-style residual reader |

The primary comparisons are `dlite_gdn` versus `dlite` and `dflash_gdn` versus
`dflash`. Each family retains its existing head, loss, and optimizer recipe;
this is not a four-way claim that DFlash and DLite have identical objectives.

## Reader and causal boundary

The fusion follows the state-reader equations from
[ReaDy](https://github.com/StartLuxLabs/ReaDy/blob/8ac61b51d529fad980566cfde7a3c41415ba93a5/ready_inference/reader.py),
with dimensions inferred from Qwen3.5 rather than its released 27B constants.
Defaults: target GDN layers **14 and 30 (zero based)**, rank 8, bottleneck 128.
Inject into draft layer **4**, after `post_attention_layernorm`, before its MLP.
The output projection starts at zero. Every query position supplies its own
read query; all positions of a block share that block's committed target state.

The reader normalizes each FP32 state by its RMS, compresses its value axis,
reads it with the draft query, appends the log RMS, and applies a residual MLP.
Native HF state layout `[B,H,K,V]` is transposed explicitly for the value/key
reader equations. Normalization is FP32; projections use the draft dtype.
Reader parameters belong to the draft and use AdamW, including under the
DLite Muon recipe. Backbones initialize before the reader, with its RNG isolated,
so paired base weights and subsequent anchor RNG streams match.

For an anchor `a`, input token `x_a` is already known to the drafter, while the
Target has executed only `x_0..x_(a-1)`. The reader receives **S_(a-1)**. It must
not receive the state after `x_a`, the end of the verified block, or the final
training answer. The bonus/correction token has not entered the Target cache.

Training uses instance-local projection hooks on just the selected native GDN
modules, without changing their output. Native qkv/a/b projections are retained
briefly. An extra causal convolution and GDN-only FP32 recurrence reconstruct
selected anchors; CUDA stores sparse snapshots directly with Triton. Native HF
still computes all target hidden states/logits. There is no full Target rerun
per anchor and no sequence-sized array of state matrices. Each batch sample uses
its own sorted anchors; right padding and masked filler anchors are supported.
Hooks are removed even when forward fails. Packed training is not supported.

Snapshots are a tuple of `[B,A,H,K,V]` tensors, one per selected target layer.
At the configured maximum of B=2, A=512 and two 32-head 128x128 layers they occupy
**4 GiB per rank**; actual A depends on valid answer positions. Reader work is
chunked into 16-anchor groups and activation-checkpointed. FSDP root input casts
are disabled to retain detached snapshots in FP32. TensorBoard records
`memory/gdn_anchor_mib` and `memory/peak_allocated_gib`.

Inference reads the live **committed** native cache before drafting, after the
previous accepted-prefix transaction. It neither owns another history KV nor
retains training anchor snapshots. State read/compression is computed per block;
it is included in the draft latency. Native single-request HF inference and the
existing static batch evaluator support these checkpoints. This does not add a
SGLang/vLLM backend or continuous batching.

GDN is a recurrent summary of the joint visual/text sequence, not a pure visual
memory. Compare image-task improvements with GSM8K before claiming visual gains.
The training follows each family's existing direct SFT; it does not reproduce
ReaDy's separate reader-only/LoRA training stages or use its published weights.

## Default training and automatic evaluation

Run from this branch, using the existing working VLM environment:

```bash
bash tests/train_gdn_ablation.sh
```

The script fixes **physical GPU 4,5,6,7**, four torchrun processes. It trains
`dflash`, `dlite`, `dflash_gdn`, `dlite_gdn` sequentially, all from scratch.
Default data is the existing
`data/finevision_qwen35_10w_img1024_relaxed/train.jsonl`; answers are used as
provided. No new dataset download, conversion, or answer regeneration occurs.
`TRAIN_DATA`, `TARGET_MODEL`, `PYTHON_BIN`, and `DLITE_WORKSPACE` accept explicit
path overrides. It does not install or upgrade the working environment.

| Setting | Default |
| --- | --- |
| Epochs | 3 |
| Per-GPU batch | 2 |
| Gradient accumulation | 8 |
| World size / global batch | 4 / 64 |
| Draft depth / block size | 5 / 8 |
| Maximum anchors | 512 |
| Seed | 42 |
| Image tokens | 1024 maximum, native processor resizing |
| Logging | TensorBoard, step progress and ETA |

Gradient accumulation is doubled from the previous eight-GPU mb2/acc4 recipe
to preserve global batch 64. The first arm validates records; later arms reuse
its exact eligible set. A fresh run group prevents checkpoint overwrites.
Inherited MAX_STEPS/MAX_SAMPLES and unrelated validation reuse are cleared.
Before training, every participating GPU runs `check_gdn_reader_cuda.py`, which
checks sparse recurrence against the CPU reference and a tiny native-image
forward/backward/export/reload/rejection path. These are small engineering
checks, not task-quality or throughput measurements. Failing checks stop the run.

Only successful training exits create `training_complete.json`. After ALL FOUR
arms succeed, the script automatically invokes `evaluate_gdn_ablation.sh` on
**GPU 4–7**. A failed arm stops the chain and does not trigger evaluation.
The evaluator checks the four training configs, uses their target and image cap,
and takes every checkpoint from that run group. It does not reuse old controls.

Defaults match the memory experiment's first evaluation: 64 seeded unique
prompts per dataset, maximum 256 output tokens, B=1, `per_token`:

- GSM8K (`text-only`)
- CharXiv validation
- Flickr30k
- TextVQA
- HRBench 4K
- HRBench 8K

The existing matrix scheduler runs same-GPU/same-input AR once per cell and
reuses it across all four drafts. Output includes acceptance length, decode
throughput/speedup, memory, transaction memory, token match, and profile stages.
`draft_ms` includes the reader; `gdn_state_prepare_ms` covers access to committed
cache views, not the full reader. The separate instrumented pass does not affect
normal speed measurements. A dagger in `matrix.md` marks AR token mismatch;
raw ratios for those cells are not evidence of lossless acceleration.

Static B>1 replicates each prompt, not distinct requests. There is no claim of
online production concurrency. OOM cells remain OOM and are not downsized.

Results live under:

```text
runs/vlm_gdn_.../
  dflash/{config.yaml,effective_config.json,train.log,last,training_complete.json}
  dlite/...
  dflash_gdn/...
  dlite_gdn/...
  cuda_check_4.log ... cuda_check_7.log
  eval.log
  evaluation/{manifest.json,matrix.csv,matrix.md,jobs,...}
```

For a short end-to-end host smoke (32 candidate records, one update per arm,
two evaluation prompts, output cap 32), use a separate fresh run group:

```bash
SMOKE_TEST=1 bash tests/train_gdn_ablation.sh
```

Smoke markers explicitly record `smoke_test=true`; they are not full training.
For a later throughput matrix on the completed checkpoints:

```bash
RESULT_DIR=/absolute/path/to/run_group/evaluation_batches \
BATCH_SIZES="1 16 64" STATE_MODES="per_token compact_replay" \
bash tests/evaluate_gdn_ablation.sh /absolute/path/to/run_group
```

Resume only an interrupted evaluation with IDENTICAL settings:

```bash
RESUME=1 bash tests/evaluate_gdn_ablation.sh /absolute/path/to/run_group
```

There is no automatic resume of partially trained arms. The aligned trainer's
explicit optimizer-state resume is available if `SAVE_TRAINING_STATE=1` was set;
otherwise use a fresh group instead of evaluating partial weights.

## Validation scope

CPU tests cover native-image causal prefix/cache agreement, future-token
invariance, padded mb2 batches, the reader's equations with nonsquare K/V,
block isolation, paired initialization and anchors, original optimizer updates,
export/reload, and rejected-prefix recovery. Shell-chain tests simulate GPU
workers to check GPU assignment and success/failure gating. The Triton kernel
also passes offline SM90 PTX compilation (no device execution), and has a CPU interpreter check:

```bash
PYTHONPATH=. python -m pytest tests/test_gdn_reader.py tests/test_gdn_launchers.py tests/test_gdn_capture_compile.py -q
TRITON_INTERPRET=1 PYTHONPATH=. python tests/check_gdn_capture_interpreter.py
```

CUDA compilation/execution and real 4B acceptance, memory and speed must be
validated on the host. The startup preflight is intentionally mandatory.
