# Qwen3.5 image/text P1 + P2

This branch adds a **single-device HF reference trainer and evaluator** for
`Qwen/Qwen3.5-4B`. Start with one local image and one question per sample.
Text-only records can use the same manifest. Target weights are frozen.

For large-scale answer regeneration, use the separate [SGLang vision client and
FineVision adapter](VLM_SERVING_DATA.md). It supports bounded concurrency,
multiple GPU servers and resume without loading target GPU weights in the client.
The single-request HF generator below remains a correctness/debugging reference.

The legacy distributed/text launchers are unchanged. They pin SGLang 0.5.9,
which requires Transformers 4.57.1. The new HF path uses an independent,
locked environment with Transformers 5.20.1 instead of breaking that stack.

## Install from the repository root

```bash
git switch research/qwen35-vlm
UV_PROJECT_ENVIRONMENT="$PWD/.venv-vlm" uv sync \
  --project environments/vlm --frozen --extra cu128
```

This installs a functional PyTorch reference path without compiling CUDA
extensions. For accelerated GDN kernels, add `--extra accelerated` to the same
command after confirming toolkit compatibility. `cu130` is an alternative to
`cu128`, not an additional extra. CUDA 12.4 nvcc and a cu128 PyTorch wheel have
different minor versions; extension compilation needs a separate check.

Run from the repository root, using `.venv-vlm/bin/python`. Do not install the
root package dependencies into this environment or use its legacy launchers.

## Explicit input manifest

Each JSONL line has this schema (paths are relative to the manifest):

```json
{"id":"example-1","image":"images/example.png","question":"Describe the objects and their positions.","answer":"A target-generated response long enough to cover a candidate block."}
```

`image` is optional for text-only data. `answer` is required only for training.
Answers should be regenerated with this exact target. Split train/eval by
image identity before regeneration. No dataset is downloaded or converted by
these entry points. Oversized inputs fail with the sample ID; visual sequences
are never silently truncated. Short answers use masked partial-block losses.

To regenerate answers from an already prepared manifest:

```bash
.venv-vlm/bin/python tests/regenerate_vlm_answers.py \
  --target-model /path/to/Qwen3.5-4B --data /path/to/prompts.jsonl \
  --output tests/outputs/train32.jsonl --max-samples 32
```

## Short SFT runs

Edit `train_data`, `target_model` (a local checkpoint path is supported), and
`output_dir` in `scripts/config/vlm/sft_student.yaml`, then:

```bash
.venv-vlm/bin/python -m litespec.vlm.train \
  --config scripts/config/vlm/sft_student.yaml
```

`sft_history.yaml` trains the same-family history-aware control. Both use the
existing DLite backbone and RNN sequential head. The control has trainable
history fusion and absolute draft positions; the student has local positions
and no history fusion. This is not an identical-parameter architecture ablation.

The reference trainer uses one record per optimizer step, a few answer anchors,
CE plus probability L1, and saves draft/optimizer/RNG/progress state to `last/`.
Resume with the same config and `--resume /path/to/last`. It does not implement
P3 two-stage teacher/student distillation, DDP, or reader training.

## Evaluate and check tokens

```bash
.venv-vlm/bin/python tests/run_vlm_eval.py \
  --target-model /path/to/Qwen3.5-4B \
  --draft-model tests/outputs/vlm_sft_student/last \
  --data /path/to/eval.jsonl \
  --policy scripts/config/vlm/dlite_no_history.json \
  --max-samples 32 --max-new-tokens 128 \
  --output tests/results/student.jsonl
```

For a **history-aware** checkpoint, use `full.json`, `drop_prefill_vision.json`,
`drop_prefill_text.json`, `drop_prefill_all.json`, `decode_window_16.json`,
`decode_window_64.json`, `drop_recent_16.json`, or `mask_all_history.json`.
Compact counterparts use `_compact.json`. A student rejects nonempty history
policies. Do not label a masked history-aware model as a trained DLite student.

The evaluator warms up AR/spec paths, uses the same prompt and output limit,
writes both token lists, and exits nonzero on greedy token mismatches.
It records prefill/decode/total latency, accepted lengths, source-token counts,
physical/visible history slots, draft/verify/recovery time and peak allocated
GPU memory. First/later rounds can be read from the ordered per-round arrays.
Memory includes the resident target AND draft for both methods.

## Policy semantics

`prefill_text` and `prefill_vision` select **draft conditioning positions**;
the target always sees the full image and text. Visual start/end delimiters
belong to the visual interface. These source groups are not pure modalities:
text-position target hidden can already contain visual information.

Anchor token `x[a]` and pivot hidden `h[a-1]` are independent of history masks.
History positions satisfy `p < a-1`. `keep_last=W` retains generated positions
`a-1-W <= p < a-1`; `drop_recent=R` removes that recent range. Prefill selectors
are independent of generation-window selectors. The exact same selector is
called in training and inference.

`mask` preserves all historical condition slots and blocks the removed ones
in attention. `compact` selects before history fusion and K/V projection;
permanently dropped prefill and expired sliding-window source states are not
retained in the reference history bank. Temporarily masked recent states are
retained in `drop_recent`, since they become eligible later. Students never
construct a history bank. Current-block K/V and target caches are unaffected.

## Correctness and performance boundaries

Target multimodal RoPE positions and GDN cache updates stay in the native HF
model. On rejection, the reference generator restores a complete target-cache
snapshot and replays only accepted inputs. This includes convolution states,
recurrent states and attention KV. Snapshot/replay overhead is counted, but
is intentionally conservative and can dominate real-model speed/memory.
The history control also reprojects selected source hidden each round; it is
not an optimized cached DFlash/DSpark serving baseline.

Use this entry point for P1/P2 correctness and acceptance diagnosis. It cannot
establish high-concurrency speedup. Those results require a later serving
implementation with matched cache recovery and scheduling. No 4B GPU result
is implied by passing tiny CPU-model tests.

## Tests

```bash
.venv-vlm/bin/python -m pytest tests/test_vlm.py tests/test_vlm_training.py -q
DLITE_REPO_PATH=/absolute/path/to/DLite .venv-vlm/bin/python -m pytest \
  tests/test_vlm_training.py -q
```

Tests use a real tiny Qwen3.5 image encoder and hybrid language model, require
no weights/dataset downloads, and cover image influence, accepted/rejected
verification, history-source windows, compact/mask output and gradient
equivalence, frozen-target training, checkpoint restore and cross-repo loading.
The common runtime and selector have mirrored copies in the DLite fork; keep
them synchronized when changing their contract.
