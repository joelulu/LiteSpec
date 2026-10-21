"""Compare AR and named checkpoints on identical prompts under reference concurrency."""
import argparse
import csv
import gc
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from litespec.vlm.baselines import load_draft
from litespec.vlm.benchmark import RequestPositionState, run_closed_loop, summarize
from litespec.vlm.context import ContextPolicy
from litespec.vlm.runtime import generate, generate_steps, load_target, prepare_example, read_records


def mismatch_details(actual, expected):
    if actual == expected:
        return None
    index = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
    return dict(position=index, ar=expected[index:index+8], spec=actual[index:index+8])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--checkpoint", nargs=2, action="append", required=True, metavar=("LABEL", "PATH"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-split", choices=["train", "validation", "smoke", "unspecified"], default="unspecified")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--verification", choices=["block", "sequential"], default="block")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution", choices=["reference", "native_text"], default="reference")
    parser.add_argument("--report-raw-speedup", action="store_true",
                        help="report observed speed ratios and continue despite token mismatches")
    args = parser.parse_args()
    if args.max_samples < 1 or args.max_new_tokens < 1 or args.warmup < 0:
        parser.error("positive sample/output limits and nonnegative warmup required")
    if any(c < 1 or c > args.max_samples for c in args.concurrency):
        parser.error("concurrency must be between 1 and max-samples")
    if args.execution == "native_text" and args.concurrency != [1]:
        parser.error("native_text currently supports concurrency 1 only")
    if args.report_raw_speedup and args.verification != "block":
        parser.error("raw performance reporting requires block verification")
    labels = [label for label, _ in args.checkpoint]
    if len(set(labels)) != len(labels) or "ar" in labels:
        parser.error("checkpoint labels must be unique and must not be 'ar'")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in labels):
        parser.error("checkpoint labels may contain only letters, digits, underscores and hyphens")
    checkpoints = [(label, str(Path(path).expanduser().resolve())) for label, path in args.checkpoint]
    for label, path in checkpoints:
        if not (Path(path) / "config.json").is_file():
            parser.error(f"missing checkpoint {label}: {path}")
    data = Path(args.data).resolve()
    records = list(read_records(data, args.max_samples))
    if args.execution == "native_text" and any(record.get("image") for record in records):
        parser.error("native_text requires a manifest without image fields")
    if len(records) != args.max_samples or len({r["id"] for r in records}) != len(records):
        parser.error("manifest must contain the requested number of samples with unique IDs")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "settings.json").write_text(json.dumps({**vars(args), "resolved_checkpoints": checkpoints,
                                                   "sample_ids": [r["id"] for r in records],
                                                   "scope": "single GPU, round-robin unbatched; excludes model loading, warmup and CPU preprocessing"}, indent=2))
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    target, processor = load_target(args.target_model, device=args.device, dtype=dtype)
    examples = [prepare_example(processor, target.config, record, device=args.device, base_dir=data.parent, training=False)
                for record in records]
    sync = (lambda: torch.cuda.synchronize(args.device)) if torch.device(args.device).type == "cuda" else (lambda: None)
    references, rows = {}, []
    canonical_ar = None
    any_failure = False
    print(f"Execution: {args.execution}; single GPU, no batched forwards.", flush=True)
    if args.report_raw_speedup:
        print("Reporting observed speed ratios regardless of token match; correctness flags remain recorded.", flush=True)
    if args.verification == "sequential":
        print("Sequential verification is a correctness diagnostic, not a speed benchmark.", flush=True)
    for label, checkpoint in [("ar", None), *checkpoints]:
        draft = load_draft(checkpoint, attn_implementation="sdpa").to(args.device, dtype).eval() if checkpoint else None
        stored = getattr(draft.config, "vlm_config", {}).get("context_policy") if draft else None
        policy = ContextPolicy(**stored) if stored else None
        def execute(example):
            if args.execution == "native_text":
                from litespec.vlm.native_text import generate_native_text
                return generate_native_text(target, draft, example, max_new_tokens=args.max_new_tokens,
                                            verification=args.verification)
            return generate(target, draft, example, max_new_tokens=args.max_new_tokens,
                            policy=policy, verification=args.verification)
        for _ in range(args.warmup):
            execute(examples[0])
        for concurrency in sorted(set(args.concurrency)):
            if torch.device(args.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(args.device)
            def factory(index, example):
                if args.execution == "native_text":
                    def one_request():
                        if False:
                            yield
                        return execute(example)
                    return one_request(), RequestPositionState(target)
                return (generate_steps(target, draft, example, max_new_tokens=args.max_new_tokens,
                                       policy=policy, verification=args.verification), RequestPositionState(target))
            print(f"Running {label}: C={concurrency}, samples={len(examples)}", flush=True)
            results, elapsed, peak = run_closed_loop(examples, factory, concurrency, synchronize=sync)
            tokens = [item["result"]["token_ids"] for item in results]
            if label == "ar":
                references[concurrency] = tokens
                if canonical_ar is None:
                    canonical_ar = tokens
                matched = sum(a == b for a, b in zip(tokens, canonical_ar))
                reference_consistent = matched == len(examples)
                speedup = 1.0
            else:
                matched = sum(a == b for a, b in zip(tokens, references[concurrency]))
                base = next(row for row in rows if row["model"] == "ar" and row["concurrency"] == concurrency)
                reference_consistent = base["valid_token_match"]
                speedup = base["wall_seconds"] / elapsed
            summary = dict(model=label, checkpoint=checkpoint, concurrency=concurrency, peak_active_requests=peak,
                           dataset_split=args.data_split, dataset_path=str(data),
                           verification=args.verification, **summarize(results, elapsed), matched=matched,
                           execution=args.execution,
                           stage_profile_available=args.execution == "reference",
                           valid_token_match=matched == len(examples) and reference_consistent,
                           ar_reference_consistent=reference_consistent, speedup_vs_ar=speedup,
                           peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 1024**3
                           if torch.device(args.device).type == "cuda" else None,
                           throughput_is_production_benchmark=False)
            any_failure |= not summary["valid_token_match"]
            expected = canonical_ar if label == "ar" else references[concurrency]
            summary["mismatch_sample_ids"] = [example.sample_id for example, actual, ref in zip(examples, tokens, expected)
                                               if actual != ref]
            summary["speedup_valid"] = summary["valid_token_match"] and args.verification == "block"
            summary["raw_wall_speedup_vs_ar"] = speedup
            summary["raw_throughput_speedup_vs_ar"] = (1.0 if label == "ar" else
                summary["output_tokens_per_second"] / base["output_tokens_per_second"])
            if not summary["speedup_valid"]:
                summary["speedup_vs_ar"] = None
            with (root / f"{label}_c{concurrency}.jsonl").open("w") as handle:
                for example, result, reference in zip(examples, results, expected):
                    handle.write(json.dumps(dict(sample_id=example.sample_id,
                                                 token_match=result["result"]["token_ids"] == reference,
                                                 first_mismatch=mismatch_details(result["result"]["token_ids"], reference),
                                                 **result)) + "\n")
            rows.append(summary)
            (root / "summary.json").write_text(json.dumps(rows, indent=2))
            with (root / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(summary), flush=True)
            del results
        del draft
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    column = "raw_tok_speedup" if args.report_raw_speedup else "speedup"
    print(f"\nmodel         C  match  accept_len  output_tok/s  latency_s  {column}", flush=True)
    for row in rows:
        acceptance = f"{row['mean_committed_length']:.3f}" if row["mean_committed_length"] is not None else "-"
        ratio = row["raw_throughput_speedup_vs_ar"] if args.report_raw_speedup else row["speedup_vs_ar"]
        speedup = f"{ratio:.3f}x" if ratio is not None else "INVALID"
        print(f"{row['model']:<13} {row['concurrency']:>1}  {row['matched']:>2}/{row['requests']:<2}  "
              f"{acceptance:>10}  {row['output_tokens_per_second']:>12.2f}  {row['mean_latency_seconds']:>9.3f}  {speedup}")
    print(f"Results: {root}", flush=True)
    if any_failure and args.report_raw_speedup:
        print("Token mismatches retained in JSONL; raw speed ratios reported as requested.", flush=True)
    elif any_failure:
        raise SystemExit("Token mismatches found; performance rows are diagnostic only. Inspect per-request JSONL.")


if __name__ == "__main__":
    main()
