"""Check experiment source integrity before allocating GPU memory (stdlib only)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    "tests/probe_dflash_visual_video.sh",
    "tests/check_dflash_context_cuda.py",
    "tests/run_dflash_context_ablation.sh",
    "litespec/vlm/dflash_context.py",
    "tests/dflash_context_config.py",
    "tests/train_dflash_context.sh",
    "tests/evaluate_dflash_context.py",
    "tests/evaluate_dflash_context.sh",
    "tests/probe_dflash_visual.py",
    "tests/test_dflash_context.py",
    "litespec/vlm/baselines.py",
    "litespec/modeling/draft/dflash.py",
    "litespec/core/attention_masks.py",
    "litespec/core/_dflash_common.py",
    "tests/pivot_feedback_probe.py",
    "tests/diagnose_pivot_feedback.py",
    "tests/diagnose_pivot_feedback.sh",
    "tests/test_pivot_joint.py",
    "litespec/vlm/pivot_augmentation.py",
    "litespec/vlm/pivot_training.py",
    "litespec/vlm/pivot_cache.py",
    "litespec/vlm/pivot_sft.py",
    "tests/train_pivot_augmentation.sh",
    "tests/evaluate_pivot_augmentation.sh",
    "tests/test_pivot_augmentation.py",
    "tests/test_pivot_protocols.py",
    "tests/pivot_screen_config.py",
    "litespec/core/dlite.py",
    "litespec/modeling/draft/dlite.py",
    "litespec/vlm/runtime.py",
    "litespec/vlm/static_batch.py",
    "litespec/vlm/train.py",
    "litespec/vlm/train_aligned.py",
    "litespec/vlm/sw_token.py",
    "tests/benchmark_video.py",
    "tests/benchmark_vlm_static_batch.py",
    "tests/evaluate_sw_token.py",
    "tests/evaluate_sw_token.sh",
    "tests/train_sw_token.sh",
    "tests/check_sw_token_source.py",
    "tests/test_sw_token.py",
    "tests/test_sw_token_entrypoints.py",
    "tests/test_sw_token_source.py",
    "tests/sw_token_probe.py",
    "tests/diagnose_sw_token.py",
    "tests/diagnose_sw_token.sh",
    "tests/train_sw_token_pair.sh",
    "tests/test_sw_token_probe.py",
)

def check_file(path):
    text = path.read_text(encoding="utf-8")
    marker = "tokens " + "truncated"
    if marker in text:
        raise ValueError(f"Truncated tool output found in source: {path}")
    if path.suffix == ".py":
        ast.parse(text, filename=str(path))
    return len(text.encode("utf-8"))

def check_repository(root=ROOT):
    return sum(check_file(Path(root) / name) for name in PATHS)

if __name__ == "__main__":
    count = check_repository()
    print(f"SW source integrity OK: {len(PATHS)} files, {count} bytes", flush=True)

