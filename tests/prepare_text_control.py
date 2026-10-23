"""Seeded reservoir sample; confirmed first-user mapping via the repo converter."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from litespec.vlm.serving_data import file_digest, output_lock, read_jsonl


def sample_rows(source, count, seed):
    if count < 1:
        raise ValueError("sample count must be positive")
    rng, selected, scanned = random.Random(seed), [], 0
    for line, row in read_jsonl(source):
        turns = row.get("conversations")
        if not isinstance(turns, list) or not turns or not isinstance(turns[0], dict):
            raise ValueError(f"line {line}: missing conversations[0]")
        question = turns[0].get("content")
        if turns[0].get("role") != "user" or not isinstance(question, str) or not question.strip():
            raise ValueError(f"line {line}: expected a nonempty first user text")
        if row.get("images") or row.get("image") or "<image>" in question:
            raise ValueError(f"line {line}: expected text-only data")
        # Do not retain source answers or later turns, even in intermediate files.
        item = dict(source_line=line, source_id=row.get("id"),
                    conversations=[dict(role="user", content=question)])
        scanned += 1
        if len(selected) < count:
            selected.append(item)
        else:
            index = rng.randrange(scanned)
            if index < count:
                selected[index] = item
    if len(selected) != count:
        raise ValueError(f"requested {count} rows but source has only {scanned}")
    rng.shuffle(selected)
    return selected, scanned


def prepare(source, directory, count=20000, seed=42):
    source, directory = Path(source).expanduser().resolve(), Path(directory).resolve()
    identity = dict(source=str(source), source_sha256=file_digest(source), samples=count,
                    seed=seed, prompt_path="conversations.0.content", turn_mode="first")
    with output_lock(directory):
        metadata, manifest = directory / "selection.json", directory / "prompts.jsonl"
        if metadata.exists():
            old = json.loads(metadata.read_text())
            if old["identity"] != identity or file_digest(manifest) != old["manifest_sha256"]:
                raise ValueError("source/selection changed; choose a fresh PREPARE_OUTPUT")
            return old
        if manifest.exists() or (directory / "sampled_source.jsonl").exists():
            raise ValueError("incomplete preparation; choose a fresh PREPARE_OUTPUT")
        converter = ROOT / "scripts/data/convert_dataset.py"
        adapter = ROOT / "scripts/data/adapters/perfectblend_first.json"
        subprocess.run([sys.executable, str(converter), "inspect", "--input", str(source),
                        "--rows", "5"], check=True)
        selected, scanned = sample_rows(source, count, seed)
        raw = directory / "sampled_source.jsonl"
        raw.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        def convert(output, limit=None):
            command = [sys.executable, str(converter), "convert", "--input", str(raw),
                       "--adapter", str(adapter), "--turn-mode", "first", "--output", str(output)]
            if limit is not None:
                command += ["--limit", str(limit)]
            subprocess.run(command, check=True)
        preview_path = directory / "preview_first5.jsonl"
        convert(preview_path, 5)
        for _, row in read_jsonl(preview_path):
            assert len(row["conversations"]) == 1 and row["conversations"][0]["role"] == "user"
        converted = directory / "converted.jsonl"
        convert(converted)
        records = list(read_jsonl(converted))
        if len(records) != count or file_digest(source) != identity["source_sha256"]:
            raise ValueError("conversion count differs or source changed")
        with manifest.open("x", encoding="utf-8") as handle:
            for (_, row), original in zip(records, selected):
                question = row["conversations"][0]["content"]
                key = hashlib.sha256(f"{identity['source_sha256']}:{original['source_line']}".encode()).hexdigest()
                handle.write(json.dumps(dict(id=key, question=question, split="train",
                    source_line=original["source_line"], source_id=original["source_id"],
                    data_source="perfectblend"), ensure_ascii=False) + "\n")
        result = dict(identity=identity, scanned=scanned, selected=count, split="all_train",
                      manifest_sha256=file_digest(manifest))
        metadata.write_text(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir, args.samples, args.seed), indent=2))
