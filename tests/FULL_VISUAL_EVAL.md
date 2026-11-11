# Full correct/wrong/mask image benchmark

Run `bash tests/evaluate_full_visual.sh` with the HF VLM Python selected by
`PYTHON_BIN`. It finds the latest experiment-1
`runs/dflash_exp1_10w_5ep_*/eval_all_b1_n32/manifest.json`, resolves the exact Full
checkpoint recorded there, and reuses its frozen 32 inputs for each image set:
CharXiv, TextVQA, Flickr30k, HR-Bench 4K and HR-Bench 8K.

The controller validates checkpoint/Target fingerprints and source-input hashes
before starting model processes. It never retrains a model. Set `SOURCE_EVAL_DIR`
to a specific prior `eval_all_b1_n32` directory if automatic selection is not
appropriate. The selected source and Full checkpoint are printed before launch.

The default condition schedule assigns all 15 dataset/condition cells across
`VISUAL_EVAL_GPUS="0 1 2 3 4 5 6 7"`, with up to eight active cells and one
active process per GPU. Every cell runs AR and Full on its own GPU. Cross-GPU
conditions do not share AR timings; each reported ratio uses its local AR.
All three conditions still use the same 32 prompts and image preprocessing.
Defaults: B=1, max_new_tokens=256, original seed/image budget, SDPA/per_token,
warmup=1. This parallelism is across independent cells; B remains 1.

Set `VISUAL_EVAL_SCHEDULING=dataset` to keep each dataset's three conditions
sequentially on the same GPU and share Correct's AR baseline through the strict
same-GPU/same-input checks. This mode has at most five active GPUs.
`FULL_VISUAL_RESULT_DIR` can specify a fresh output directory. `--dry-run`
validates and freezes inputs/commands without loading models.

These are **Draft visual-context interventions**. Target always sees the
correct image. Wrong replaces visual-position Target hidden with features from
a different-image donor, using the existing deterministic donor alignment;
own text hidden and absolute positions stay unchanged. Donors and their IDs are
recorded by the benchmark. Mask blocks Draft attention to visual positions.
This does not remove visual information already encoded in text hidden.

Wrong is a diagnostic condition: its extra donor prefill is excluded from model
timing. All 32 samples contribute to throughput/acceptance metrics regardless
of AR match. AR match is reported separately. There is no generation-path
alignment between conditions; use the existing `probe_dflash_visual.py` entry
for fixed-Target-prefix/anchor diagnostics when required.

Outputs: `manifest.json`, `inputs/`, per-condition `logs/` and `results/`,
`comparison.json`, `comparison.csv`, `status.json`, and `visual_comparison.md`.
The last report contains only Full rows, with an explicit condition column.
Interrupt cleanup and GPU serialization use the existing evaluation controller.
