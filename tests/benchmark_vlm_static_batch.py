"""Offline replicated-input static batch benchmark; throughput excludes prefill."""
import argparse
import csv
import gc
from importlib.util import find_spec
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from litespec.vlm.baselines import load_draft
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.runtime import load_target, prepare_example, read_records
from litespec.vlm.static_batch import generate_static_batch


def summarize_batches(results, batch_size):
    tokens = sum(row["output_tokens"] for row in results)
    prefill = sum(row["prefill_seconds"] for row in results)
    decode = sum(row["decode_seconds"] for row in results)
    total = sum(row["total_seconds"] for row in results)
    lengths = [length for row in results for length in row["accept_lengths"]]
    return dict(batch_size=batch_size, concurrency=batch_size, unique_prompts=len(results),
        requests=len(results)*batch_size, output_tokens=tokens,
        output_tokens_per_second=tokens/decode, decode_tokens_per_second=tokens/decode,
        end_to_end_tokens_per_second=tokens/total,
        prefill_seconds=prefill, decode_seconds=decode, wall_seconds=total,
        mean_latency_seconds=total/len(results),
        mean_committed_length=sum(lengths)/len(lengths) if lengths else None,
        mean_accepted_draft_tokens=sum(n-1 for n in lengths)/len(lengths) if lengths else None,
        speculative_rounds=len(lengths), recovery_replays=sum(row["recovery_replays"] for row in results),
        target_decode_calls=sum(row["target_decode_calls"] for row in results),
        mean_input_tokens=sum(row["prefill_tokens"] for row in results)/len(results),
        scheduler="static_replicated_batch", benchmark_mode="static_replicated_batch",
        timing_scope="decode_only_after_prefill", throughput_is_production_benchmark=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--checkpoint", nargs=2, action="append", required=True, metavar=("LABEL", "PATH"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-split", choices=["train", "validation", "smoke", "unspecified"], default="unspecified")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=["auto", "sdpa", "eager", "flash_attention_2"], default="auto")
    parser.add_argument("--verification", choices=["block"], default="block")
    parser.add_argument("--report-raw-speedup", action="store_true", help="accepted for compatibility; observed ratios are always reported")
    args = parser.parse_args()
    if min([args.max_samples, args.max_new_tokens, *args.batch_sizes]) < 1 or args.warmup < 0:
        parser.error("positive sample/token/batch limits and nonnegative warmup required")
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)) or "ar" in labels:
        parser.error("checkpoint labels must be unique and not ar")
    if any(not label or any(not (c.isalnum() or c in "-_") for c in label) for label in labels):
        parser.error("checkpoint labels must contain only letters/digits/hyphens/underscores")
    checkpoints = [(label, str(Path(path).expanduser().resolve())) for label, path in args.checkpoint]
    for label, path in checkpoints:
        if not (Path(path)/"config.json").is_file():
            parser.error(f"missing checkpoint {label}: {path}")
    data = args.data.resolve()
    records = list(read_records(data, args.max_samples))
    if len(records) != args.max_samples or len({record["id"] for record in records}) != len(records):
        parser.error("requested number of unique samples required")
    attention = args.attention_backend
    if attention == "auto":
        attention = "flash_attention_2" if torch.device(args.device).type == "cuda" and find_spec("flash_attn") else "sdpa"
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = {**vars(args), "data": str(data), "output_dir": str(root),
        "resolved_checkpoints": checkpoints, "sample_ids": [row["id"] for row in records],
        "actual_attention_backend": attention,
        "benchmark_mode": "static_replicated_batch", "timing_scope": "decode_only_after_prefill"}
    (root/"settings.json").write_text(json.dumps(settings, default=str, indent=2))
    dtype = torch.float32 if torch.device(args.device).type == "cpu" else torch.bfloat16
    target, processor = load_target(args.target_model, device=args.device, dtype=dtype, attention=attention)
    examples = [prepare_example(processor, target.config, record, device=args.device,
                               base_dir=data.parent, training=False) for record in records]
    rows, references = [], {}
    print(f"True static replicated batches: {sorted(set(args.batch_sizes))}; attention={attention}", flush=True)
    print("Decode-only throughput; prefill reported separately. No continuous batching. Match does not gate ratios.", flush=True)
    for label, checkpoint in [("ar", None), *checkpoints]:
        draft = load_draft(checkpoint, attn_implementation=attention).to(args.device, dtype).eval() if checkpoint else None
        for batch_size in sorted(set(args.batch_sizes)):
            def run(example, cap):
                with RequestPositionState(target).activate():
                    return generate_static_batch(target, draft, example, batch_size=batch_size, max_new_tokens=cap)
            for _ in range(args.warmup):
                run(examples[0], min(16, args.max_new_tokens))
            if torch.device(args.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(args.device)
            print(f"Running {label}: B={batch_size}, {len(examples)} unique prompts / {len(examples)*batch_size} sequences", flush=True)
            results = [run(example, args.max_new_tokens) for example in examples]
            tokens = [result["token_ids"] for result in results]
            if label == "ar":
                references[batch_size] = tokens
            expected = references[batch_size]
            matched = sum(a == b for a, b in zip(tokens, expected))*batch_size
            summary = dict(model=label, checkpoint=checkpoint, dataset_split=args.data_split,
                dataset_path=str(data), attention_backend=attention, **summarize_batches(results, batch_size))
            base = summary if label == "ar" else next(row for row in rows if row["model"] == "ar" and row["batch_size"] == batch_size)
            summary.update(matched=matched, valid_token_match=matched == summary["requests"],
                raw_throughput_speedup_vs_ar=summary["output_tokens_per_second"]/base["output_tokens_per_second"],
                raw_wall_speedup_vs_ar=base["wall_seconds"]/summary["wall_seconds"],
                raw_end_to_end_speedup_vs_ar=summary["end_to_end_tokens_per_second"]/base["end_to_end_tokens_per_second"],
                mismatch_sample_ids=[example.sample_id for example, a, b in zip(examples, tokens, expected) if a != b],
                peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/1024**3
                    if torch.device(args.device).type == "cuda" else None)
            with (root/f"{label}_b{batch_size}.jsonl").open("w") as handle:
                for example, result, reference in zip(examples, results, expected):
                    handle.write(json.dumps(dict(sample_id=example.sample_id, token_match=result["token_ids"] == reference,
                                                 result=result)) + "\n")
            rows.append(summary)
            (root/"summary.json").write_text(json.dumps(rows, indent=2))
            with (root/"summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(summary), flush=True)
        del draft
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    print("\nmodel          B  match     accept_len  decode_tok/s  prefill_s  decode_s  speedup", flush=True)
    for row in rows:
        length = row["mean_committed_length"]
        length = f"{length:.3f}" if length is not None else "-"
        print(f"{row['model']:<14} {row['batch_size']:>2} {row['matched']:>3}/{row['requests']:<3} {length:>10} "
              f"{row['output_tokens_per_second']:>13.2f} {row['prefill_seconds']:>10.2f} {row['decode_seconds']:>9.2f} "
              f"{row['raw_throughput_speedup_vs_ar']:>8.3f}x", flush=True)
    print(f"Results: {root}", flush=True)


if __name__ == "__main__":
    main()
