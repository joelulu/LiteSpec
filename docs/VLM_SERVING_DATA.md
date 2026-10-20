# FineVision regeneration through SGLang

This is **answer regeneration only**. It loads the HF processor on CPU, sends
real images through the OpenAI vision API, and writes training-compatible
`id/image/question/answer/answer_target` records. It does not load GPU weights,
produce target hidden states, train drafts, or change the existing HF path.

## 1. Start a target server in a separate environment

Use an existing SGLang environment that supports the Qwen3.5 multimodal model.
Do not install SGLang into `.venv-vlm`: the legacy root package and the VLM HF
environment have different dependency requirements. Check that environment's
`python -m sglang.launch_server --help` before running the launcher. The server
launcher deliberately does not install or upgrade packages.

From the LiteSpec repository root, in a separate terminal:

```bash
SGLANG_PYTHON=/absolute/path/to/sglang-environment/bin/python \
CUDA_VISIBLE_DEVICES=7 PORT=30000 \
TARGET_MODEL=/home/jovyan/LMM/lmm_model/Qwen3.5-4B \
bash scripts/data/serve_qwen35_vlm.sh
```

Replace the Python placeholder with your actual SGLang interpreter. Defaults:
one GPU (`TP_SIZE=1`), BF16, localhost, served name `qwen35-vlm`, context 4096,
and static memory fraction 0.80. Additional server flags may be appended to
the launcher. Wait until the server is ready. A port accepting connections
alone does not establish multimodal correctness; run the two-image smoke below.

## 2. Prepare and regenerate

Use `.venv-vlm` for the client. It requires no new GPU libraries; `tqdm` is
already a Transformers dependency. Defaults match the user's existing local
FineVision and Qwen3.5-4B paths; override `SOURCE_DATA` and `TARGET_MODEL` as needed.

```bash
# First verify two real image requests.
MAX_SAMPLES=2 CONCURRENCY=1 \
bash scripts/data/prepare_finevision_serving.sh http://127.0.0.1:30000

# Then measure a 64-candidate run, starting at concurrency 8.
MAX_SAMPLES=64 CONCURRENCY=8 \
bash scripts/data/prepare_finevision_serving.sh http://127.0.0.1:30000

# Once validated, regenerate 10,000 candidate records.
MAX_SAMPLES=10000 CONCURRENCY=8 \
bash scripts/data/prepare_finevision_serving.sh http://127.0.0.1:30000
```

The adapter inspects the first five records and uses the confirmed mapping
`conversations[0].content`, turn mode `first`. Later turns are ignored. Only
single-image records with existing local files are accepted. The candidate
limit applies **before deduplication and token-length checks**, so saved answers
can be fewer than the limit. Questions retain their original text after removal
of the single `<image>` marker. No images are resized or rewritten by this script.
The native processor still applies its usual model preprocessing.

The adapter and split IDs match the earlier HF script: question+resolved image
path SHA256 for IDs, resolved image path hash for approximately 5% validation.
Different questions about the same resolved path always have the same split.
Copies of an image at different paths are not content-deduplicated.

Each limit has its own directory beside the LiteSpec checkout, e.g.
`../data/finevision_qwen35_sglang_10000/`:

- `manifest/prompts.jsonl`, metadata, first-five inspection and adapter skips;
- `train.jsonl` and `val.jsonl`, each flushed after every successful record;
- `skipped.jsonl`: permanent local input/length validation skips;
- `errors.jsonl`: failed requests or output validation; automatically retried
  on the next run (historical errors are retained);
- `generation_config.json` and `last_summary.json`: provenance and this run's
  successful sample/token throughput, including CPU validation and HTTP cost.

Repeat the identical command to resume. Changing concurrency or server addresses
does not invalidate data. Changing the manifest/model/generation settings does;
use a new `PREPARE_OUTPUT`. Do not run two clients against one output directory;
an exclusive writer lock rejects that. A malformed partial JSONL tail is reported
explicitly; it is never silently truncated. No HF output files are overwritten
or imported. To compare concurrency fairly, use separate output directories with
the same manifest; resuming a completed run does not measure generation speed.

The client sends `temperature=0`, `chat_template_kwargs.enable_thinking=false`
and `MAX_NEW_TOKENS` (default 256). It reserves that budget plus 32 template tokens
within `MAX_LENGTH` (default 4096), checks final training length, prefix stability
and nonempty supervision. `finish_reason=length` answers are retained and marked
in generation metadata. The check uses the **local** native HF processor; keep
server weights, tokenizer, template and image preprocessing matched to this
target. A served model alias check cannot verify that two servers loaded
identical weight files. SGLang and HF greedy outputs need not be token-identical.

## 3. Multiple GPUs

Only start services on GPUs you intend to use. Launch one full 4B replica per
GPU in separate terminals, each with a different port. For example, **if GPUs 6
and 7 are available**, launch the server script with `CUDA_VISIBLE_DEVICES=6
PORT=30001` and `CUDA_VISIBLE_DEVICES=7 PORT=30000`, respectively. Then use one client:

```bash
MAX_SAMPLES=10000 CONCURRENCY=8 \
bash scripts/data/prepare_finevision_serving.sh \
  http://127.0.0.1:30000 http://127.0.0.1:30001
```

Concurrency is per address (8 each, 16 total in this example); outstanding work
is bounded. Results are written by the main client thread. Failover is not
automatic: every configured server must pass `/v1/models`, and transient errors
retry against the assigned server. Authentication can be provided through
`SGLANG_API_KEY` in the client and matching server arguments. For localhost
requests behind a proxy, set `NO_PROXY=127.0.0.1,localhost` as appropriate.

## Standalone manifest and generator

The bash wrapper is optional. The CPU-only adapter can run before a server exists:

```bash
.venv-vlm/bin/python scripts/data/prepare_finevision_manifest.py \
  --input /path/to/finevision_sharegpt.jsonl \
  --output-dir /path/to/manifest --max-samples 10000

.venv-vlm/bin/python scripts/data/regenerate_vlm_serving.py \
  --manifest /path/to/manifest/prompts.jsonl --output-dir /path/to/answers \
  --target-model /path/to/Qwen3.5-4B --model qwen35-vlm \
  --server-address http://127.0.0.1:30000 --concurrency 8
```

The generator exits nonzero if any submitted record failed. Fix failures and
rerun to retry them. Neither GPU utilization nor a particular speedup is promised:
measure after server warmup; image processing, input lengths and concurrency all
affect throughput. New data can be used by the existing VLM trainer; this change
does not add a DFlash multimodal trainer.

## Verification

```bash
.venv-vlm/bin/python -m pytest tests/test_vlm_serving.py -q
```

CPU tests use a local HTTP server to verify image bytes, first-turn selection,
image-level split, deduplication, multiple-server concurrency, real HTTP retries,
resume, output locks, length checks and incompatible run identity. They require
no SGLang/GPU/model downloads. Real Qwen3.5 server and GPU throughput verification
must be run on the training server.
