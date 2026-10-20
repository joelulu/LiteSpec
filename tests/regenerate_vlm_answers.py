"""Regenerate answers for the explicit VLM manifest; no dataset conversion."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litespec.vlm.runtime import generate, load_target, prepare_example, read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    source = Path(args.data).resolve()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    target, processor = load_target(args.target_model, device=args.device,
                                    dtype=torch.float32 if args.device == "cpu" else torch.bfloat16)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in read_records(source, args.max_samples):
            example = prepare_example(processor, target.config, row, device=args.device,
                                      base_dir=source.parent)
            result = generate(target, None, example, max_new_tokens=args.max_new_tokens)
            record = dict(row)
            if record.get("image"):
                path = Path(record["image"])
                record["image"] = str(path if path.is_absolute() else source.parent / path)
            record["answer"] = processor.tokenizer.decode(result["token_ids"], skip_special_tokens=True)
            record["answer_target"] = args.target_model
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"id": record["id"], "answer_tokens": len(result["token_ids"])}), flush=True)


if __name__ == "__main__":
    main()
