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

Each dataset runs Correct, Wrong, then Mask on the same physical GPU. Correct
runs AR and Full; Wrong/Mask reuse Correct's AR outputs and timings through the
existing strict same-GPU/same-input reference checks. Defaults: B=1, 32 samples,
max_new_tokens=256, original seed/image budget, SDPA/per_token, warmup=1.
`VISUAL_EVAL_GPUS="4 5 6 7"` is the default; five datasets are queued without
overlapping jobs on a GPU. `FULL_VISUAL_RESULT_DIR` can specify a fresh output
directory. `--dry-run` validates and freezes inputs/commands without loading models.

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
