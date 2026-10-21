"""Audit the first failed text request from a previous comparison (not a benchmark).

Replays native block, native sequential and reference block generation. Each
cached multi-token target call is compared with tokenwise calls from an identical
cache snapshot, without loading a second target or changing returned outputs.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import benchmark_vlm_checkpoints as benchmark
from litespec.vlm.runtime import read_records


def cache_tensors(cache):
    tensors = {}
    def visit(value, path):
        if torch.is_tensor(value):
            tensors[path] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for key, item in enumerate(value):
                visit(item, f"{path}.{key}")
    for index, layer in enumerate(getattr(cache, "layers", [])):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            visit(getattr(layer, name, None), f"layer{index}.{name}")
    return tensors


def cache_difference(left, right):
    a, b = cache_tensors(left), cache_tensors(right)
    differences = []
    for name in sorted(a.keys() | b.keys()):
        if name not in a or name not in b or a[name].shape != b[name].shape:
            differences.append(dict(name=name, shape_mismatch=True))
            continue
        if a[name].numel():
            delta = float((a[name].float() - b[name].float()).abs().max())
            if delta:
                differences.append(dict(name=name, max_abs=delta))
    return sorted(differences, key=lambda row: row.get("max_abs", float("inf")), reverse=True)[:8]


def audit_target(target, handle, tokenizer=None):
    forward = target.forward
    slots = [(module, "rope_deltas") for module in target.modules() if hasattr(module, "rope_deltas")]
    calls = 0
    @torch.no_grad()
    def wrapped(*args, **kwargs):
        nonlocal calls
        ids = kwargs.get("input_ids", args[0] if args else None)
        cache = kwargs.get("past_key_values")
        # Exclude prompt prefill: native prefill supplies an empty cache.
        eligible = (ids is not None and ids.shape[-1] > 1 and cache is not None
                    and cache.get_seq_length() > 0)
        if not eligible:
            return forward(*args, **kwargs)
        calls += 1
        length_before = cache.get_seq_length()
        before = copy.deepcopy(cache)
        rope_before = [getattr(module, name) for module, name in slots]
        result = forward(*args, **kwargs)
        rope_after = [getattr(module, name) for module, name in slots]
        block_cache = result.past_key_values
        logits, details = [], []
        try:
            for (module, name), value in zip(slots, rope_before):
                setattr(module, name, value)
            for index in range(ids.shape[-1]):
                one_kwargs = dict(kwargs)
                one_kwargs.update(input_ids=ids[:, index:index+1], past_key_values=before, logits_to_keep=1)
                for key in ("position_ids", "cache_position"):
                    if key in one_kwargs and one_kwargs[key] is not None:
                        one_kwargs[key] = one_kwargs[key][..., index:index+1]
                one = forward(**one_kwargs)
                before = one.past_key_values
                logits.append(one.logits[:, -1:])
            sequential = torch.cat(logits, 1).float()
            block = result.logits.float()
            # Recovery may request logits_to_keep=1, but still has a multi-token input.
            sequential = sequential[:, -block.shape[1]:]
            offset = ids.shape[-1] - block.shape[1]
            for index in range(block.shape[1]):
                b, s = block[0, index], sequential[0, index]
                bt, st = int(b.argmax()), int(s.argmax())
                top = s.topk(2)
                detail = dict(input_position=length_before + offset + index,
                              block_next=bt, sequential_next=st, token_equal=bt == st,
                              max_abs_logit_delta=float((b-s).abs().max()),
                              sequential_top2_margin=float(top.values[0]-top.values[1]),
                              sequential_preference_over_block=float(s[st]-s[bt]))
                if tokenizer is not None and bt != st:
                    detail.update(block_text=tokenizer.decode([bt]), sequential_text=tokenizer.decode([st]))
                details.append(detail)
            row = dict(call=calls, cache_length_before=int(length_before), input_ids=ids[0].tolist(),
                       checked_positions=details,
                       cache_differences=cache_difference(block_cache, before))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        finally:
            for (module, name), value in zip(slots, rope_after):
                setattr(module, name, value)
        return result
    target.forward = wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True,
                        help="group directory containing text_native/settings.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    stage = args.previous_run / "text_native"
    settings = json.loads((stage / "settings.json").read_text())
    previous = [json.loads(line) for line in (stage / "dlite_original_c1.jsonl").read_text().splitlines()]
    failed = next((row for row in previous if (row["sample_id"] == args.sample_id if args.sample_id else not row["token_match"])), None)
    if failed is None:
        parser.error("no matching failed request found")
    sample_id = failed["sample_id"]
    record = next(row for row in read_records(settings["data"]) if row["id"] == sample_id)
    if record.get("image"):
        parser.error("this diagnostic currently requires text")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir.resolve() / "sample.jsonl"
    manifest.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Auditing {sample_id}; previous first mismatch: {failed.get('first_mismatch')}", flush=True)
    print("Correctness audit only; timings include extra target calls and cache copies.", flush=True)
    original_load = benchmark.load_target
    summaries = []
    for name, execution, verification in (("native_block", "native_text", "block"),
                                           ("native_sequential", "native_text", "sequential"),
                                           ("reference_block", "reference", "block")):
        trace_path = args.output_dir / f"{name}_target_audit.jsonl"
        with trace_path.open("w") as handle:
            def load(*a, **kw):
                target, processor = original_load(*a, **kw)
                audit_target(target, handle, getattr(processor, "tokenizer", None))
                return target, processor
            benchmark.load_target = load
            sys.argv = ["benchmark", "--target-model", settings["target_model"],
                        "--checkpoint", "dlite_original", dict(settings["resolved_checkpoints"])["dlite_original"],
                        "--data", str(manifest), "--output-dir", str(args.output_dir / name),
                        "--max-samples", "1", "--max-new-tokens", str(settings["max_new_tokens"]),
                        "--concurrency", "1", "--warmup", "0", "--execution", execution,
                        "--verification", verification, "--device", args.device]
            try:
                benchmark.main()
            except SystemExit as error:
                if error.code not in (None, 0, 1) and not str(error.code).startswith("Token mismatches found;"):
                    raise
            finally:
                benchmark.load_target = original_load
        summary_path = args.output_dir / name / "summary.json"
        rows = json.loads(summary_path.read_text())
        if len(rows) != 2:
            raise RuntimeError(f"{name} failed before producing the draft result")
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        differences = [dict(call=row["call"], **position) for row in trace
                       for position in row["checked_positions"] if not position["token_equal"]]
        summary = dict(case=name, token_match=rows[-1]["valid_token_match"],
                       first_output_mismatch=json.loads((args.output_dir / name / "dlite_original_c1.jsonl").read_text())["first_mismatch"],
                       audited_chunk_calls=len(trace), chunk_token_disagreements=len(differences),
                       first_chunk_disagreement=differences[0] if differences else None)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    def tokens(case, model):
        return json.loads((args.output_dir / case / f"{model}_c1.jsonl").read_text())["result"]["token_ids"]
    canonical = tokens("native_sequential", "ar")
    sequential_spec = tokens("native_sequential", "dlite_original")
    for summary in summaries:
        summary["ar_matches_sequential_case"] = tokens(summary["case"], "ar") == canonical
        summary["draft_matches_sequential_case"] = tokens(summary["case"], "dlite_original") == sequential_spec
    (args.output_dir / "diagnosis.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print("Audit timing is diagnostic only: extra cache copies and target forwards invalidate performance measurements.")


if __name__ == "__main__":
    main()
