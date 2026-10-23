# Serving the current LiteSpec multimodal checkpoints

Source audit: 2026-10-23. This is a deployment assessment, not a claim that the
user's installed SGLang/vLLM versions or checkpoints have passed GPU serving.
No serving environment is installed/upgraded by the matrix benchmark.

## Upstream support vs checkpoint compatibility

| Draft | SGLang main | vLLM main | Current checkpoints |
|---|---|---|---|
| DFlash | Native model/worker, hybrid Target state handling | Native Qwen3 DFlash model/proposer, explicit Qwen3.5 multimodal support | Strong candidate for loading/config/weight-map validation; not yet GPU-validated |
| DSpark | Native model/worker; vanilla/gated/RNN head variants | Native Qwen3 DSpark model/speculator; inspected model builds its Markov head directly | Verify actual head variant, config parser, confidence options and accepted-window convention |
| DLite | No matching native implementation found in inspected model/worker paths | No matching native implementation found in inspected model/worker paths | Requires history-free pivot/CHS adapter + sequential head + worker integration |

Official DLite README still lists SGLang/vLLM integrations as unchecked roadmap
items. A Qwen3.5 Target supported by an engine does not automatically make a
custom Draft supported. Renaming `architectures` alone is not a valid adapter.

Current LiteSpec DFlash uses `DFlashDraftModel`, per-layer Q/K/V projections,
`fc`, `hidden_norm`, Target-shared embeddings/lm_head and selected Target hidden
layers. SGLang's loader contains the corresponding packed-QKV/gate-up mapping
and architecture entry. This makes SGLang a sensible first integration target.
Loading must still check that every required parameter was consumed: the
upstream loader may skip unexpected names, so "server started" is insufficient.

LiteSpec DSpark can use vanilla/gated/RNN heads. SGLang's RNN implementation
contains the same `joint_proj` three-way gate/candidate/output structure as our
RNN variant. The existing early experiment's effective checkpoint config, not
the set of supported variants, decides which head must be instantiated. The
inspected vLLM Qwen3 DSpark path directly constructs its own Markov head;
do not presume it handles every LiteSpec recurrent-head variant unchanged.

DLite is more than a DFlash checkpoint with history disabled: it uses pivot/CHS
conditioning, local query positions and a sequential vocabulary head. Serving
integration needs model registration/loading, per-request pivot extraction and
commit, draft-history-free scheduling, packed variable-length requests, and
CUDA-graph-safe buffers/head execution. Qwen3.5 MRoPE and hybrid Target states
must remain under the engine's own management.

## Multimodal and deployment path

Image preprocessing/vision encoding remain in the Target's normal multimodal
prefill. The Draft predicts text conditioned on Target hidden features; it
need not independently run the image encoder. Validate selected hidden layer
indices/offset, MRoPE-to-Draft position handling, attention masking, mask token,
shared vocabulary, block/bonus-token convention and committed prefix indices.

Recommended progression:

1. Pin a compatible SGLang revision in a separate serving environment. Keep
   the existing Torch 2.9.1 training/HF environment unchanged.
2. DFlash, TP=1: compare converted/projected features, draft logits and accepted
   prefixes on one text and one image input. Audit complete weight loading.
3. Run six datasets with real distinct HTTP requests at concurrency 1/16/64,
   same processor budgets and generation settings as the reference baseline.
4. Adapt/validate DSpark, including Markov/confidence and state commits.
5. Add DLite's history-free model/head/worker integration and validate against
   the current HF path before claiming native high-concurrency support.
6. Test TP=2/4 or replicated TP=1 workers after single-device correctness.

For a 4B Target that fits one GPU, eight TP=1 replicas can increase aggregate
serving capacity and be exposed behind one router endpoint. A TP=8 engine
instead splits a model across eight GPUs; it has communication overhead and
does not guarantee better throughput/latency for a small model. Compare AR and
Draft at the same TP/replica configuration. Report per-request concurrency
separately from per-replica batch and total service concurrency.

The matrix script uses independent single-GPU offline jobs to finish research
comparisons faster. It does not create an HTTP service or implement engine TP.

## Inspected primary sources

- [SGLang speculative decoding docs](https://docs.sglang.ai/advanced_features/speculative_decoding.html)
- [SGLang DFlash model](https://github.com/sgl-project/sglang/blob/efd2ce8a9df027d3607766fc5e7efcb3390cf738/python/sglang/srt/models/dflash.py)
- [SGLang DSpark model](https://github.com/sgl-project/sglang/blob/efd2ce8a9df027d3607766fc5e7efcb3390cf738/python/sglang/srt/models/dspark.py)
- [SGLang DSpark config](https://github.com/sgl-project/sglang/blob/efd2ce8a9df027d3607766fc5e7efcb3390cf738/python/sglang/srt/speculative/dspark_components/dspark_config.py)
- [SGLang DFlash worker](https://github.com/sgl-project/sglang/blob/efd2ce8a9df027d3607766fc5e7efcb3390cf738/python/sglang/srt/speculative/dflash_worker_v2.py)
- [vLLM DFlash model](https://github.com/vllm-project/vllm/blob/c0a41721c72db80768cf5220047c3e7192d4352e/vllm/model_executor/models/qwen3_dflash.py)
- [vLLM DSpark model](https://github.com/vllm-project/vllm/blob/c0a41721c72db80768cf5220047c3e7192d4352e/vllm/model_executor/models/qwen3_dspark.py)
- [vLLM DFlash proposer](https://github.com/vllm-project/vllm/blob/c0a41721c72db80768cf5220047c3e7192d4352e/vllm/v1/spec_decode/dflash.py)
- [Official DLite roadmap](https://github.com/SJTU-DAI-Sys/DLite#-roadmap)
