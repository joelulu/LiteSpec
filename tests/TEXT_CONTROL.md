# PerfectBlend text-only DFlash control

Run from LiteSpec with the existing HF training Python and separate SGLang Python:

```bash
SGLANG_PYTHON="$HOME/.conda/envs/dflash/bin/python" \
LITESPEC_EAGER_BLOCK_MASK=1 \
bash tests/run_dflash_text_control.sh
```

This executes the following stages, stopping on failure:

1. Inspect five source rows. Use the confirmed `conversations.0.content` adapter,
   discard all existing answers/later turns, reservoir-sample 20,000 source rows
   uniformly over the full file using seed 42. Convert five selected rows and
   validate them before converting the full selection. Save source line/ID provenance.
2. Start eight single-GPU SGLang replicas; wait for every health/model check.
   Generate greedy Qwen3.5-4B answers with thinking disabled, max 256 output tokens,
   max 4096 total tokens. Use the same CPU template/length validation as FineVision.
   Stop only the owned services before training, including on failure/interruption.
3. Train DFlash from scratch through `scripts/dlite/run_training_vlm.sh` using its
   existing HF Qwen3.5 multimodal target adapter with text-only inputs. Defaults:
   3 epochs, block 8, micro batch 2, accumulation 4, eight workers (global batch 64),
   TensorBoard enabled. Remaining settings inherit the same DFlash VLM recipe.
4. Evaluate AR, original DLite, new text-trained DFlash and the previous multimodal
   DFlash on GSM8K and the five named VLM datasets. Defaults match the preceding
   diagnostic: static B=1, 16 prompts, max 128 new tokens, native image processing.
   Preserve token-match diagnostics; raw speed ratios do not require exact match.

The 20,000 figure counts selected **rollout candidates**, all assigned to train.
Generation failures, overlong inputs or insufficient answer blocks may reduce
actual eligible training rows. Inspect `last_summary.json`, `skipped.jsonl` and
training `effective_config.json` before comparing results. There is no automatic
refill or silent removal of malformed source records. Seeded sampling is over
rows, not deduplicated questions. Repeated questions may be selected if duplicated
in the source. This does not establish independence from GSM8K evaluation data.

Configuration:

- `SOURCE_DATA`, `TEXT_SAMPLES=20000`, `SAMPLE_SEED=42`, `PREPARE_OUTPUT`
- Serving: `DATA_GPUS=0,1,2,3,4,5,6,7`, `BASE_PORT`, `CONCURRENCY=8` **per server**,
  `MAX_NEW_TOKENS=256`, `MAX_LENGTH=4096`, `MIN_ANSWER_TOKENS=1`.
- Training: `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE` (inferred from visible list),
  `MICRO_BATCH_SIZE=2`, `ACCUMULATION_STEPS=4`, `BLOCK_SIZE=8`, `EPOCHS=3`, `OUTPUT_DIR`.
  To match the previous six-GPU global batch, set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
  and `NPROC_PER_NODE=6` (global batch 48); rollout still uses eight GPUs.
- Evaluation: `EVAL_GPU`, `BATCH_SIZES`, `EVAL_SAMPLES`, `EVAL_MAX_NEW_TOKENS`,
  `TEXT_DATA`, `IMAGE_MAX_PIXELS`, `IMAGE_MIN_PIXELS`, `DFLASH_VLM`, `RESULT_DIR`.
  `dflash_text` / `dflash_vlm` selectors also work in the standard comparison script
  with `DFLASH_TEXT` / `DFLASH_VLM` checkpoint paths.

Default data output: `../data/perfectblend_qwen35_20k_text/`.
Default training output: `../runs/dflash_text_20k_3ep_<timestamp>_<pid>/`.
`last` links to the final checkpoint; tables are under its parent run's
`evaluation_<timestamp>_<pid>/speedup_tables.md`.

Stages can run separately with `MODE=prepare`, `MODE=train`, or `MODE=eval`.
Prepare is resumable for an unchanged source/selection/generation identity.
For retrying failed HTTP requests, repeat `MODE=prepare`; already successful rows
are reused. Skipped rows remain skipped. Changing generation parameters requires
a fresh `PREPARE_OUTPUT`. Training requires a fresh `OUTPUT_DIR`; it does not resume
optimizer state. For evaluation only:

```bash
MODE=eval DFLASH_TEXT="/path/to/new/text/run/last" EVAL_GPU=6 \
bash tests/run_dflash_text_control.sh
```

Interpretation: weak results on both modalities warrant checking training and
data distribution, but do not alone prove an implementation bug. Text outperforming
the current visual checkpoint motivates a matched 20k visual control and checks of
rollout answer distribution before claiming visual training is unnecessary. This
benchmark measures speed/acceptance, not task accuracy or production concurrency.
