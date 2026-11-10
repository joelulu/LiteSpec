# Full/Pivot1 evaluation on eight GPUs

Run `bash tests/evaluate_dflash_all.sh` from the context worktree, with `RUN_GROUP`
set to the intended training directory. The wrapper chooses the HF VLM Python;
it does not use the shell's active Draft-OPD environment.

| GPU | Dataset | Unique samples |
| --- | --- | --- |
| 0 | GSM8K | 32 |
| 1 | CharXiv validation | 32 |
| 2 | TextVQA | 32 |
| 3 | Flickr30k | 32 |
| 4 | HR-Bench 4K | 32 |
| 5 | HR-Bench 8K | 32 |
| 6 | VideoDetailCaption | 16 |
| 7 | Video-MME | 16 |

Each GPU runs AR, Full and Pivot1 sequentially for its dataset, with one shared
Target per process. All eight datasets run concurrently. Defaults are static
batch size 1, maximum generation length 256, seed 42, one warmup and SDPA with
`per_token` state. The image budget is 1024 tokens; video uses 16 frames and at
most 200704 pixels per frame. Video-MME follows the existing video-description
protocol, not official question-answer accuracy. These are throughput/agreement
benchmarks, not task-answer accuracy evaluation.

Before model processes launch, the controller resolves `full/last` and
`pivot1/last` to fixed checkpoint directories, checks config/weight files, and
freezes all selected inputs. Each video has its own one-dataset selection file,
matching the video benchmark's strict snapshot format. The manifest records
GPU assignments, sample counts, commands, source paths, input hashes and model
fingerprints. Missing datasets/fields are reported together before loading any
Target weights. No source dataset is modified.

Image readers explicitly use the existing `reference` policy: read `prompt`
when present, otherwise the first conversation's `content`. Inherited
`PROMPT_FIELD=question` cannot change this policy. GSM8K uses the existing
`question`/single-`turns` reader and the existing default text data path.
`EVAL_PROMPT_FIELD` is the explicit override for the image readers.

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
export RUN_GROUP="$DLITE_WORKSPACE/runs/dflash_full_sw1_10w_2ep_mb4_acc16_20261110_114622_2746034"
export EVAL_SAMPLES=32 VIDEO_SAMPLES=16 EVAL_BATCH_SIZE=1
bash tests/evaluate_dflash_all.sh
```

Overrides: `PYTHON_BIN`, `TARGET_MODEL`, `DATA_DIR`, `IMAGE_ROOT`, `TEXT_DATA`,
`VIDEO_DATA_ROOT`, `EVAL_GPUS` (eight distinct selectors separated by spaces),
`EVAL_MAX_NEW_TOKENS`, `IMAGE_MAX_TOKENS`, `VIDEO_FRAMES`,
`VIDEO_MAX_PIXELS_PER_FRAME`, `SAMPLE_SEED`, `EVAL_PROMPT_FIELD`, and `RESULT_DIR`.
The default GSM8K manifest is
`$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main-0708-new/eval_datasets/gsm8k.jsonl`;
set `TEXT_DATA` if it lives elsewhere.

Use `bash tests/evaluate_dflash_all.sh --dry-run` to validate/freeze selections
and inspect commands without starting model processes. Use a fresh output
directory for the subsequent real run. The controller prints log paths:

```text
eval_all_.../manifest.json
eval_all_.../inputs/
eval_all_.../logs/<dataset>.log
eval_all_.../results/<dataset>/
eval_all_.../status.json
eval_all_.../comparison.json
eval_all_.../comparison.csv
eval_all_.../comparison.md
```

The combined report is refreshed as jobs complete. Failed/incomplete jobs are
marked explicitly; other datasets continue, and the overall command exits
nonzero if any dataset fails. AR token mismatches are reported without hiding
throughput measurements. Ctrl+C terminates only this controller's child process
groups. It does not kill existing training or unrelated evaluation processes.
