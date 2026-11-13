# Full / Pivot1, 300k candidates, block 8 / 16

Entry: `bash tests/run_dflash_30w.sh all`. Stages: `data`, `train`, `eval`.
All stages use GPUs **0,1,2,3**, never GPUs 4-7. The generation client and
training use `PYTHON_BIN` (the HF VLM environment). Serving uses the separate
`SGLANG_PYTHON`, defaulting to `$HOME/.conda/envs/dflash/bin/python` through the
existing pipeline. Do not install SGLang into the VLM training environment.

## Data

The existing confirmed FineVision adapter reads `conversations[0].content`,
first user turn, single image. It inspects five source records and deduplicates
image/question IDs. The limit is **300000 eligible candidate prompts**, not
300000 final training records. Too few candidates fails before training.
Image-content splitting during regeneration holds about 5% out for validation;
validation rows never enter common training data.

Four independent TP=1 SGLang replicas regenerate Target answers with greedy
decoding, thinking off, at most **256 new tokens**, max total **4096 tokens**,
and image cap **1024 tokens**. Adaptive answer budgeting uses remaining space.
Default client concurrency is 8 per server. The pipeline owns and stops only
its server/client process groups before preprocessing/training.
Matching interrupted generation in this new output directory can be retried.
Old 10w answers are not automatically spliced into a different generation run:
there is no implicit cross-run identity assumption.

Two genuine parallel CPU preflights (default 8 threads, chunk 64) use the
existing trainer rules, once with block8 and once with block16. The common
JSONL contains their intersection in original generated-record order.
Rejected reasons, source indices and both candidate-cache fingerprints are
recorded. The two final reusable caches are derived only from this checked
intersection, remap indices to the common JSONL and use its actual path/hash.
The trainer's `reuse_validation` validates their counts/IDs/index hashes before
atomic publication. They are not edited old 10w caches.

Default data directory:
`$DLITE_WORKSPACE/data/finevision_qwen35_30w_img1024_b8b16/`:

* `regenerated/`: manifests, train/val JSONL, generation identity and errors.
* `candidate_validation/`: original parallel preflight decisions for both blocks.
* `common/`: frozen `train.jsonl`, `manifest.json`, source indices/rejections,
  `validation_b8/`, `validation_b16/`.

Completed common data is verified and reused; data, source outputs, validation
metadata or Target identity changes stop reuse. Images/processors must also
remain unchanged, as required by the original cache mechanism. To recheck
changed images/processors, choose a new `DATA_ROOT`.

## Training

Four independent from-scratch Drafts train sequentially:
`full_b8`, `pivot1_b8`, `full_b16`, `pivot1_b16`. Target Qwen3.5-4B is frozen.
Each uses 4 ranks, micro batch 2, accumulation 32, global batch 256, 5 epochs,
5 Draft layers, 512 anchors, max length 4096, image cap 1024, AdamW LR 6e-4,
warmup .04, weight decay 0, clip 1, gamma 4, seed 42. Block16 supervises more
positions per anchor: this is equal data exposure, not equal compute.
Stale warm-start/debug/batch environment variables do not override this recipe.

The launcher first checks all four actual models for prefix parity, finite
loss/backward, checkpoint reload and CUDA Flex. A four-rank BF16/Flex/FSDP
SHARD_GRAD_OP smoke test exercises the production FP32-master optimizer with
two updates x 32 micro steps. These small-model tests do not load Target weights
or establish full-size memory fit. Any failure gates formal training.
Models export `epoch_1` through `epoch_5` and `last`; optimizer states are not
saved. `SUITE_RUN_GROUP` must be fresh. Effective settings, code revision,
common data hash, and all epoch exports are verified before evaluation.

## Evaluation

All four epoch-5 exports retain their own trained block size. Each GPU processes
one dataset queue at a time: GSM8K, CharXiv, TextVQA, Flickr30k, HRBench4K/8K,
VideoDetailCaption and Video-MME. **32 samples each, B=1, greedy, 256 new tokens**.
Videos use existing 16-frame / 200704 max-pixels-per-frame settings; Video-MME
remains the description protocol. A matching latest experiment-1 frozen selection
is reused automatically when found; `SOURCE_EVAL_DIR` can select one explicitly.
Otherwise raw
datasets use the existing reference prompt policy and seed42.

MathVista+GovReport uses existing `LONGCTX_DATA_DIR`, short/8k/32k/64k,
32 samples, repeat1, with smoke-gated formal evaluation. No download is started
by the training launcher. Both evaluation data sources are checked before
formal training. All models share identical long-context inputs.
If a suite fails, the other suite is still attempted and the command exits
nonzero. Checkpoint exports remain available for `eval` reruns.

Reports include accepted Draft length, Committed (including Target token),
First hit, decode tok/s, same-GPU AR ratio, matches and peak memory. /Full uses
the Full control with the SAME block size; AR rows use Full block8. All samples
contribute irrespective of AR matches. Long-context reports add prefill/total
timings and separate Draft profiling.
Image/text request JSONL and summary JSON additionally save per-position verifier
agreement and contiguous-prefix acceptance (position1..7 or1..15). Later unary
agreement is conditioned on the proposed prefix, not a gold Target trajectory;
it is not task accuracy. Counts include evaluated candidate positions before
EOS trimming and skip nonexistent positions in final truncated blocks.
Verifier masks are reduced on CPU after timing; no diagnostic GPU reductions
or host transfers are inserted into timed decode rounds.

To repeat evaluation:
`RUN_GROUP=/path/to/completed/group bash tests/run_dflash_30w.sh eval`.
CPU tests: `python -m unittest discover -s tests -p test_dflash_30w.py`;
numeric: `python tests/check_dflash_blocks.py --device cpu`.
