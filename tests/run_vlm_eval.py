"""Compare real HF AR and DLite outputs on an explicit image/text JSONL manifest."""

import argparse
import json
import sys
from pathlib import Path

import torch

# Permit `python tests/run_vlm_eval.py` without a global editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.vlm.runtime import generate, load_target, prepare_example, read_records
from litespec.vlm.context import ContextPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--policy", help="JSON context policy; default comes from checkpoint")
    parser.add_argument("--output", default="tests/results/vlm_eval.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    target, processor = load_target(args.target_model, device=args.device, dtype=dtype)
    draft = DLiteDraftModel.from_pretrained(args.draft_model, attn_implementation="sdpa").to(args.device, dtype).eval()
    stored = getattr(draft.config, "vlm_config", {}).get("context_policy")
    policy = ContextPolicy(**json.loads(Path(args.policy).read_text())) if args.policy else (
        ContextPolicy(**stored) if stored else ContextPolicy.no_history()
    )
    data_path = Path(args.data).resolve()
    records = list(read_records(data_path, args.max_samples))
    if not records:
        raise ValueError("evaluation manifest is empty")
    examples = [prepare_example(processor, target.config, row, device=args.device,
                                base_dir=data_path.parent) for row in records]
    for _ in range(args.warmup):
        generate(target, None, examples[0], max_new_tokens=args.max_new_tokens)
        generate(target, draft, examples[0], max_new_tokens=args.max_new_tokens, policy=policy)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            results = []
            for model in (None, draft):
                if torch.device(args.device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(args.device)
                results.append(generate(target, model, example, max_new_tokens=args.max_new_tokens, policy=policy))
            ar, spec = results
            equal = ar["token_ids"] == spec["token_ids"]
            failures += not equal
            result = {"sample_id": example.sample_id, "token_match": equal, "ar": ar, "spec": spec,
                      "text": processor.tokenizer.decode(spec["token_ids"], skip_special_tokens=True)}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"sample_id": example.sample_id, "token_match": equal,
                              "accept_lengths": spec["accept_lengths"]}), flush=True)
    if failures:
        raise SystemExit(f"{failures} token mismatches; inspect {path}")


if __name__ == "__main__":
    main()
