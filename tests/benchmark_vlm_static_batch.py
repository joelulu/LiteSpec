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
from litespec.vlm.runtime import load_target, prepare_example
from litespec.vlm.static_batch import generate_static_batch
from vlm_benchmark_data import read_eval_records


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
    parser.add_argument("--checkpoint", nargs=2, action="append", default=[], metavar=("LABEL", "PATH"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-split", choices=["train", "validation", "smoke", "unspecified"], default="unspecified")
    parser.add_argument("--dataset-name", default="unspecified")
    parser.add_argument("--reference-input", action="store_true", help="read prompt or first conversations.content, sample with seed")
    parser.add_argument("--input-mode", choices=["text", "vision"], default="vision")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--prompt-field", choices=["reference", "prompt", "question", "first-content"], default="reference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-max-pixels", type=int)
    parser.add_argument("--image-min-pixels", type=int)
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
    if min([args.max_new_tokens, *args.batch_sizes]) < 1 or args.max_samples < (0 if args.reference_input else 1) or args.warmup < 0:
        parser.error("positive sample/token/batch limits and nonnegative warmup required")
    if any(value is not None and value < 1 for value in (args.image_min_pixels, args.image_max_pixels)):
        parser.error("image pixel limits must be positive")
    if args.image_min_pixels and args.image_max_pixels and args.image_min_pixels > args.image_max_pixels:
        parser.error("image minimum must not exceed maximum")
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)) or "ar" in labels:
        parser.error("checkpoint labels must be unique and not ar")
    if any(not label or any(not (c.isalnum() or c in "-_") for c in label) for label in labels):
        parser.error("checkpoint labels must contain only letters/digits/hyphens/underscores")
    checkpoints = [(label, str(Path(path).expanduser().resolve())) for label, path in args.checkpoint]
    for label, path in checkpoints:
        if not (Path(path)/"config.json").is_file():
            parser.error(f"missing checkpoint {label}: {path}")
    data = args.data.expanduser().resolve()
    inspection = None
    if args.reference_input:
        from reference_vlm_data import load_reference_records
        records, inspection = load_reference_records(data, args.max_samples, args.seed, args.image_root,
                                                     args.input_mode == "vision", args.prompt_field)
    else:
        records = list(read_eval_records(data, args.max_samples))
    if not records or (not args.reference_input and len(records) != args.max_samples) or len({record["id"] for record in records}) != len(records):
        parser.error("requested number of unique samples required")
    attention = args.attention_backend
    if attention == "auto":
        attention = "flash_attention_2" if torch.device(args.device).type == "cuda" and find_spec("flash_attn") else "sdpa"
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = {**vars(args), "data": str(data), "output_dir": str(root),
        "resolved_checkpoints": checkpoints, "sample_ids": [row["id"] for row in records],
        "actual_attention_backend": attention,
        "benchmark_mode": "static_replicated_batch", "timing_scope": "decode_only_after_prefill",
        "data_inspection": inspection}
    (root/"settings.json").write_text(json.dumps(settings, default=str, indent=2))
    dtype = torch.float32 if torch.device(args.device).type == "cpu" else torch.bfloat16
    target, processor = load_target(args.target_model, device=args.device, dtype=dtype, attention=attention)
    if args.image_max_pixels is not None or args.image_min_pixels is not None:
        from reference_vlm_image_budget import configure_image_budget
        configure_image_budget(processor, args.image_min_pixels, args.image_max_pixels)
    image_size = getattr(getattr(processor, "image_processor", None), "size", None)
    settings["effective_image_processor_size"] = dict(image_size) if image_size is not None else None
    (root/"settings.json").write_text(json.dumps(settings, default=str, indent=2))
    examples = [prepare_example(processor, target.config, record, device=args.device,
                               base_dir=data.parent, training=False) for record in records]
    input_metadata = []
    for record, example in zip(records, examples):
        grid = example.inputs.get("image_grid_thw")
        image_id = getattr(target.config, "image_token_id", None)
        patch_size = getattr(getattr(target.config, "vision_config", None), "patch_size", None)
        metadata = dict(sample_id=example.sample_id, source_id=record.get("source_id"),
            source_row=record.get("source_row"), image_path=record.get("image"),
            original_image_size=record.get("original_image_size"),
            input_tokens=example.prefill_length,
            visual_tokens_with_delimiters=int(example.vision_mask.sum()),
            image_tokens=int((example.inputs["input_ids"] == image_id).sum()) if image_id is not None else 0,
            processed_image_hw=[[int(row[1])*patch_size, int(row[2])*patch_size] for row in grid.tolist()]
                if grid is not None and patch_size is not None else None,
            image_grid_thw=grid.tolist() if grid is not None else None)
        input_metadata.append(metadata)
    (root/"inputs.json").write_text(json.dumps(input_metadata, indent=2))
    if args.reference_input:
        print("First processed input: " + json.dumps(input_metadata[0]), flush=True)
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
                dataset_path=str(data), dataset_name=args.dataset_name,
                input_mode=args.input_mode if args.reference_input else "from_manifest",
                mean_visual_tokens=sum(meta["visual_tokens_with_delimiters"] for meta in input_metadata)/len(input_metadata),
                mean_image_tokens=sum(meta["image_tokens"] for meta in input_metadata)/len(input_metadata),
                attention_backend=attention, **summarize_batches(results, batch_size))
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
                    decode = getattr(getattr(processor, "tokenizer", None), "decode", None)
                    handle.write(json.dumps(dict(sample_id=example.sample_id, token_match=result["token_ids"] == reference,
                        decoded_text=decode(result["token_ids"], skip_special_tokens=True) if callable(decode) else None,
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
