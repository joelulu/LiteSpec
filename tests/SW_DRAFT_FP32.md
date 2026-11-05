# Draft-only FP32 alignment check

This is an untimed TextVQA/early-20k diagnostic, not a speed benchmark or training.
Use the same parent and sample protocol as the previous diagnostic. On CUDA,
Target remains BF16. Draft is first rounded to BF16 as before and only then
promoted to FP32: checkpoint weight rounding is held constant. Frozen Target
hidden values and Target lookup outputs are cast at the Draft input boundary.
The shared Target embedding module/weights are never converted or updated.
FP32 matmul precision is set to highest and CUDA TF32 is disabled for this check.

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
cd "$DLITE_WORKSPACE/LiteSpec-video-sw-token"
git fetch origin refs/heads/video-sw-token:refs/remotes/origin/video-sw-token
git merge --ff-only origin/video-sw-token
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
export RESULT_DIR="$DLITE_WORKSPACE/runs/sw_textvqa_draft_fp32_$(date +%Y%m%d_%H%M%S)_$$"
export REUSE_PREFIXES_FROM="$DLITE_WORKSPACE/runs/sw_textvqa_diagnosis_20261104_235440_3773815"
EVAL_GPU=7 DRAFT_DTYPE=float32 EVAL_SAMPLES=16 EVAL_MAX_NEW_TOKENS=256 \
PROBE_STRIDE=8 PROBES_PER_SAMPLE=16 \
bash tests/diagnose_sw_token.sh --alignment-atol 0.001 --alignment-rtol 0.0001
cat "$RESULT_DIR/diagnostics.md"
```

Reuse verifies the prior selection hash, unchanged parent identity, Target path,
Target precision, image budget/effective processor size, seed, sample count,
output cap, anchor stride/count, and each regenerated AR token sequence, prefill
length and probe count. It does not reselect samples. If any Target AR sequence
changes, it writes `prefix_mismatch.json` and stops: the precision comparison
would otherwise be confounded. This checks identical token prefixes; Target
hidden values are regenerated rather than loaded from the previous run.

The report prints both model dtypes, whether prior AR prefixes were verified,
and previous/current masked-control error maxima. The FP32 command uses stricter
numeric tolerances than the previous BF16 check. Compare the deltas, not only
PASS/FAIL, because the tolerances differ. A large reduction supports a numeric
precision explanation; a persistent difference needs further mask/position or
attention-path inspection. FP32 alone does not uniquely identify the source.
A failed alignment writes the report and exits with code 2. There is no automatic
training or threshold relaxation.

With no `DRAFT_DTYPE` override, the existing BF16 diagnostic is preserved on GPU.
For default CPU tests the Target is FP32, as before; use the mixed-BF16/FP32 tiny
unit test to validate that the Target remains unmodified.

The aligned training entrypoint is also repaired to honor `init_checkpoint`,
`context_window_size`, and the shared chunked K=1/K=2 training control. It keeps
fresh optimizer warm starts separate from optimizer resume. Model-only exports
remain the default, and experiment launchers keep `SAVE_TRAINING_STATE=0`.
No full-checkpoint GPU run or training result is claimed by these code tests.
