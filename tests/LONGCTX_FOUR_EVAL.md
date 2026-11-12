# Four-GPU long multimodal evaluation

`tests/evaluate_longctx_four.sh` runs only **AR, Full, Pivot1**. Defaults:
16 fixed samples, seed 42, greedy, max_new_tokens 256, static B=1, SDPA,
GDN `per_token`, checkpoint `epoch_5` from the formal 10w-candidate 5-epoch group.
No training or checkpoint modification occurs.

| Physical GPU | Dataset | Variants |
|---|---|---|
| 0 | MathVista + GovReport | short, 8192, 32768, 65536 actual input tokens |
| 1 | Video-MME | 32, 64, 128 native video frames |
| 2 | VideoDetailCaption | 32, 64, 128 native video frames |
| 3 | MMLongBench-Doc | complete document pages; measured natural length |

GPU assignment can be changed by `EVAL_GPUS="0 1 2 3"`. Fewer GPUs serialize
dataset queues; no tensor parallelism or online concurrency is implied.
Each variant first runs all three models on its first two samples. Only a
successful smoke proceeds to the full 16-sample cell. `SMOKE_SAMPLES=0` disables
this gate. IDs match across variants and models. A run uses a fresh result
directory; it does not overwrite completed preparation or evaluation files.

## Local sources

Default source root: `$DLITE_WORKSPACE/data/multimodal_longctx100`.
MMLongBench raw inputs:

```
mmlongbench_doc/train-00000-of-00001.parquet
mmlongbench_doc/documents/*.pdf
```

Video sources reuse `$HOME/LMM/liyt/datasets/eval_bench/{Video-MME,VideoDetailCaption}`.
MathVista first reuses the verified existing
`$DLITE_WORKSPACE/data/longctx_mathvista100`, or the prepared
`multimodal_longctx100/mathvista-govreport` pool. Otherwise supply local
`--mathvista-parquet` and `--govreport-parquet`, or put the original shards in
`multimodal_longctx100/{mathvista,govreport}/` (also searches `data/longctx_sources`).
`--mathvista-source` overrides the already-prepared MathVista pool.

The wrapper defaults to `HF_HUB_OFFLINE=1`; sources must be local. It does not
download missing data. CPU preparation occurs inside each assigned worker before
model loading. PDF pages are rendered to 896px longest edge and capped at 1024
visual tokens per page. All pages remain in order. The document adapter never
passes answer/evidence metadata into model input, nor inserts the PDF text layer.
There is no document-page truncation to meet a token budget. Video frames use
max 448x448 pixels each; frame counts have independent uniform sampling.
Video-MME retains the existing **description** task, not official MCQ accuracy.

## Run from the worktree root

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
export RESULT_DIR="$DLITE_WORKSPACE/runs/longctx_four_n16_$(date +%Y%m%d_%H%M%S)_$$"
EVAL_GPUS="0 1 2 3" EVAL_SAMPLES=16 EVAL_MAX_NEW_TOKENS=256 \
  bash tests/evaluate_longctx_four.sh
```

The default group is
`runs/dflash_exp1_10w_5ep_mb4_acc8_20261110_154705_2794634`.
`RUN_GROUP` and `CHECKPOINT_NAME` override it; there is no silent fallback from
missing `epoch_5` to `last`. Fingerprints and context types are checked before
launch. All models must have matching block size. `--tests Video-MME ...` can
select a subset; mapping follows the selected order.

Outputs: `manifest.json`, `status.json`, four dataset logs, frozen `inputs/`,
per-case `results/`, smoke outputs, `comparison.md` and `.csv`.
Acceptance is **committed length including the guaranteed Target token**;
accepted draft count is also in CSV. Speedups are raw observed same-case ratios.
Match is recorded, not a gate for reporting throughput. Model end-to-end timing
includes prefill + decode but excludes model loading, CPU preprocessing and
host transfers, matching the existing benchmark convention. Prefill seconds are
summed across the sampled requests. Image/document per-request processed metadata
is in `results/*/inputs.json`; video metadata is `results/*/*_inputs.json`.
Context-limit violations fail explicitly before generation. GPU OOM or malformed
input fails its dataset queue, is visible in status/logs, and produces a nonzero
final exit; completed cells in other queues are still reported. Reports ignore
partial summary files and require all three model rows for a cell.

```bash
tail -f "$RESULT_DIR/logs/MMLongBench-Doc.log"
cat "$RESULT_DIR/comparison.md"
```

CPU tests validate command budgets, shared smoke/full snapshots, multi-page
routing and report handling. GPU throughput and complete real datasets must be
validated on the user's server; CPU tests are not GPU benchmark evidence.

## Long-prefill memory mitigation

The four-GPU runner now explicitly requests `--prefill-hidden-mode selected` and
`--sample-memory`. Standalone benchmark defaults remain `all` for backwards
compatibility; pass these two flags to opt in when retrying a frozen cell.
Transformers' native layer-list capture retains only requested decoder outputs
for prefill (not all language layers or the embedding output). The selected
last layer follows the same final-norm convention. Decode/verification math,
GDN state scheme, images/pages and sample selection are unchanged. For bounded
Pivot/recent contexts, slice the last window BEFORE stacking layer features.

`sample_memory.jsonl` writes start/complete/failed events for each timed request,
with sample ID/index, input length, page count (documents), visible physical GPU
IDs, and allocated/reserved/peak memory. Events and peak-counter resets occur
outside model timing; no additional synchronizations are inserted inside decode.
Per-model peak reports take the maximum across all sampled requests. Failed
requests are re-raised and never silently skipped. These statistics do not by
themselves establish a memory leak or guarantee that every long document fits.

The shell wrapper defaults to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
when not already set. This can help fragmentation but does not replace reducing
live tensor memory. No `empty_cache()` call is inserted in timed generation.
Chunked prefill is not introduced: it needs separate MRoPE/GDN/cache equivalence
validation. No page/token-budget reduction is used as an OOM fallback.

CPU tests compare dense/selected hidden values (including final norm), logits,
generated tokens and acceptance lengths using a tiny real Qwen3.5, image/text,
Full/Pivot1 and per_token/replay state modes. Real GPU FLA peak reduction and the
previously failing document must still be measured on the user's machine.
