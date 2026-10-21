"""Evaluation reader matching the user's prompt/first-content/image script.

No conversion or source rewrite. First conversation content is an explicit
single-turn policy, matching the supplied reference script.
"""
import argparse
import json
from pathlib import Path
import random

DATASETS = ("charxiv_val", "flickr30k", "textvqa", "hr_bench_4k", "hr_bench_8k")
DATASET_FILES = dict(charxiv_val="charxiv_val.jsonl", flickr30k="flickr30k_test.jsonl",
                    textvqa="textvqa_test.jsonl", hr_bench_4k="hr_bench_4k.jsonl", hr_bench_8k="hr_bench_8k.jsonl")


def dataset_path(directory, name):
    if name not in DATASET_FILES:
        raise ValueError(f"unknown dataset: {name}; choose {' '.join(DATASETS)}")
    directory = Path(directory).expanduser().resolve()
    # Prefer the explicit evaluation split even when a legacy filename exists.
    path = directory / DATASET_FILES[name]
    if path.is_file():
        return path
    legacy = directory / f"{name}.jsonl"
    if legacy.is_file():
        return legacy
    available = ', '.join(p.name for p in sorted(directory.glob('*.jsonl')))
    raise FileNotFoundError(f"Dataset not found: {path}; available JSONL files: {available}")


def resolve_image_path(value, image_root):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image must be one non-empty local path string")
    path = Path(value.replace("\\/", "/").replace("\\", "/")).expanduser()
    root = Path(image_root).expanduser().resolve()
    if path.is_absolute():
        return path.resolve()
    if root.name == "imgs" and path.parts and path.parts[0] == "imgs":
        path = Path(*path.parts[1:])
    return (root/path).resolve()


def get_prompt(record, field="reference"):
    source = field
    if field == "reference":
        source = "prompt" if record.get("prompt") else "first-content"
    if source in ("prompt", "question"):
        prompt = record.get(source)
    elif source == "first-content":
        conversations = record.get("conversations")
        prompt = conversations[0].get("content") if isinstance(conversations, list) and conversations and isinstance(conversations[0], dict) else None
    else:
        raise ValueError(f"unsupported prompt field: {field}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"missing string {source}; available keys: {sorted(record)}")
    return prompt, source


def load_reference_records(path, max_samples=16, seed=42, image_root=None, require_image=True,
                           prompt_field="reference"):
    from PIL import Image
    path = Path(path).expanduser().resolve()
    if max_samples < 0:
        raise ValueError("sample count must be nonnegative (0 means all)")
    raw = []
    with path.open() as handle:
        for index, line in enumerate(handle):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{index+1}: expected a JSON object")
                raw.append((index, record))
    if not raw:
        raise ValueError(f"empty dataset: {path}")
    selected = random.Random(seed).sample(raw, max_samples) if 0 < max_samples < len(raw) else raw
    root = Path(image_root).expanduser() if image_root is not None else path.parent/"imgs"
    def normalize(pair):
        index, source = pair
        prompt, field = get_prompt(source, prompt_field)
        record = dict(id=f"{path.stem}:{index}", question=prompt, source_id=source.get("id"),
                      source_row=index, source_prompt_field=field)
        if require_image:
            image_path = resolve_image_path(source.get("image"), root)
            if not image_path.is_file():
                raise FileNotFoundError(f"{path.name}:{index+1}: image not found: {image_path}")
            with Image.open(image_path) as image:
                record.update(image=str(image_path), original_image_size=list(image.size))
        return record
    records = [normalize(pair) for pair in selected]
    first = [normalize(pair) for pair in raw[:5]]
    inspection = dict(dataset=path.stem, source=str(path), total_records=len(raw),
        selected_records=len(records), seed=seed, prompt_policy=prompt_field,
        source_rows=[record["source_row"] for record in records],
        first_five=[{**{key: value for key, value in record.items() if key != "question"},
                     "prompt_preview": record["question"][:240]} for record in first])
    return records, inspection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resolve-dataset", choices=DATASETS, help="print the selected evaluation file and exit")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", choices=["text", "vision"], default=["text", "vision"])
    parser.add_argument("--prompt-field", choices=["reference", "prompt", "question", "first-content"], default="reference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.data_dir.expanduser()
    if args.resolve_dataset:
        print(dataset_path(directory, args.resolve_dataset))
        return
    inspections = []
    for name in args.datasets:
        if name not in DATASETS:
            parser.error(f"unsupported dataset {name}; choose {' '.join(DATASETS)}")
        _, info = load_reference_records(dataset_path(directory, name), args.samples, args.seed,
            args.image_root, "vision" in args.modes, args.prompt_field)
        inspections.append(info)
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(inspections, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
