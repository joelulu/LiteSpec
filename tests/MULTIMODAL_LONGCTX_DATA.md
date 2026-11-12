# Four long-context evaluation pools (at most 100 questions each)

Preparation only; no training, rollout, or Target weight loading.

| Pool | Inputs | Selection / variants |
|---|---|---|
| `mathvista-govreport` | Original MathVista image + query; unrelated GovReport filler | Existing fixed MathVista IDs, first <=100; short / 8192 / 32768 / 65536 actual Qwen3.5 input tokens after encoding |
| `Video-MME` | Local native video + existing fixed description prompt | Seeded duration-stratified unique videos; 32 / 64 / 128 frames, same IDs |
| `VideoDetailCaption` | Local native video + original `question` | Seeded unique videos; same frame variants |
| `MMLongBench-Doc` | Every PDF page in order + original `question` | Seeded round-robin across documents; up to 100 questions, answers/evidence metadata excluded from model input |

Video-MME retains the repository's description-speed protocol. It is not an
accuracy evaluation on the official multiple-choice questions. No subtitles or
synthetic conversation history are added. Each frame count uses the existing
uniform-index native-video processor independently: sample IDs match, frame
index sets are not promised to be nested. Short videos can repeat sampled frames;
encoding records their exact frame indices and timestamps.

MMLongBench-Doc uses `yubo2333/MMLongBench-Doc`, annotation
`data/train-00000-of-00001.parquet`, selected `documents/{doc_id}` only.
The resolved Hub revision is pinned for annotation and PDF downloads within a
run. PDFs are rendered once per document, longest edge 896 px by default, and
all pages are retained. No evidence-based page selection, PDF text layer, or
answer hint enters inputs. An unanswerable question remains in the sample pool.

## 1. Prepare data (CPU)

Run from repository root with the existing VLM Python:

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
# Install only missing preparation dependencies; do not upgrade torch/transformers.
"$PYTHON_BIN" -m pip install pandas pyarrow huggingface_hub Pillow PyMuPDF

PREPARE_SAMPLES=100 SAMPLE_SEED=42 \
LONGCTX_SUITE_DIR="$DLITE_WORKSPACE/data/multimodal_longctx100" \
bash tests/prepare_multimodal_longctx.sh
```

Local videos are reused from `$HOME/LMM/liyt/datasets/eval_bench/`.
Override via `--video-root`. Existing MathVista/GovReport Parquet files can be
supplied using `--mathvista-parquet` and `--govreport-parquet`. Existing document
sources use `--doc-parquet /path/train.parquet --doc-root /path/documents`.
Without these, selected document PDFs and the two image/text shards download
through the normal Hugging Face cache. `HF_ENDPOINT` is respected by the Hub.

Prepare selected subsets independently:

```bash
bash tests/prepare_multimodal_longctx.sh --tests Video-MME VideoDetailCaption
bash tests/prepare_multimodal_longctx.sh --tests MMLongBench-Doc --doc-page-edge 896
```

The requested sample count is a cap for videos/documents, not permission to
silently skip malformed records or missing media. Missing local videos fail with
their paths. MathVista uses the fixed reference pool (100 available IDs).

Each pool gets `base.jsonl`, a manifest and first-five inspection. Videos get
`frames_32.jsonl`, `frames_64.jsonl`, `frames_128.jsonl`. Documents get ordered
`images` and `page_numbers`, plus rendered `pages/`. `suite.json` indexes prepared
subsets. Settings and file hashes are checked on reuse; failed conversions do
not publish partial pools. Use a fresh output directory to change settings.
Do not launch concurrent writers for the same output pool.

## 2. Encode / count actual inputs (optional, CPU)

```bash
PYTHONPATH=. "$PYTHON_BIN" tests/encode_multimodal_longctx.py \
  --data-dir "$DLITE_WORKSPACE/data/multimodal_longctx100" \
  --target-model "$HOME/LMM/lmm_model/Qwen3.5-4B" \
  --image-max-tokens 1024 --max-new-tokens 256
```

Requires `av` in the existing VLM environment for videos. No model weights are
loaded. Exact MathVista length fitting and hundreds of video decodes can take
substantial CPU time. Use `--tests MMLongBench-Doc` etc. for separate runs.

Outputs under `encoded/{pool}/`: MathVista `inputs/{short,ctx8192,ctx32768,ctx65536}.jsonl`
plus `input_lengths.json`; video/document `input_lengths.jsonl` with measured
input/visual counts, length bin, per-sample metadata and context-limit flags.
1024 is a **per-page/image cap**, not a whole-document cap. Video uses the
per-frame pixel budget (default 448x448), not the image-token cap.
Document/page and video lengths are measured, not forced to 8K/32K/64K.
Over-limit video/documents remain in the audit as flagged records; they are
never silently truncated. GPU memory requirements are not inferred from token
counts alone. MathVista refuses an exact target length that cannot fit the
original task or the model context limit.

The existing single-image benchmark cannot directly consume document `images`:
use `document_example` from `encode_multimodal_longctx.py` for evaluation-side
multi-page processing. Never feed these records through a reader that ignores
`images`. Native video evaluation must pass each row's `num_frames` and
`max_pixels_per_frame` to `video_runtime.prepare_video`. Preparation does not
automatically launch a benchmark.
