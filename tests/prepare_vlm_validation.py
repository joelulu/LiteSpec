"""Prepare reusable VLM eligibility before loading Target weights (Linux).

Run via prepare_vlm_validation.sh to select the HF VLM environment. Imports of
the training dependencies query CUDA, so keep one GPU visible even though all
image preparation runs on CPU. --help does not import those dependencies.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace


REUSE_KEYS = (
    "data_sha256", "train_data", "target_model", "max_length", "block_size",
    "debug_max_samples", "image_max_tokens", "pivot_two_rounds",
)


def load_dependencies():
    import torch
    from tqdm import tqdm
    from transformers import AutoConfig, AutoProcessor
    from litespec.vlm import train_aligned
    from litespec.vlm.sw_token import configure_image_tokens
    torch.set_num_threads(1)
    return SimpleNamespace(
        train=train_aligned, AutoConfig=AutoConfig, AutoProcessor=AutoProcessor,
        configure=configure_image_tokens, tqdm=tqdm,
    )


def signature(cfg):
    return {k: cfg.get(k, False if k == "pivot_two_rounds" else None) for k in REUSE_KEYS}


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def find_cache(records, cfg, roots, train):
    seen = set()
    for root in roots:
        # A forced recheck should supersede older caches with the same settings.
        files = sorted(Path(root).rglob("effective_config.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for config_file in files:
            directory = config_file.parent.resolve()
            if directory in seen or any(p.startswith(".building-") for p in directory.parts):
                continue
            seen.add(directory)
            if not (directory / "skipped_training.jsonl").is_file():
                continue
            try:
                previous = json.loads(config_file.read_text())
                if not isinstance(previous, dict):
                    raise ValueError("validation metadata is not an object")
                if signature(previous) != signature(cfg):
                    continue
                train.reuse_validation(records, cfg, directory)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                print(f"Skipping unusable validation cache {directory}: {exc}", flush=True)
                continue
            return directory
    return None


def parallel_validate(records, cfg, target_config, deps, workers, chunk_size):
    local = threading.local()

    def initialize():
        local.processor = deps.AutoProcessor.from_pretrained(
            cfg["target_model"], local_files_only=True,
        )
        if cfg.get("image_max_tokens") is not None:
            deps.configure(local.processor, target_config, int(cfg["image_max_tokens"]))

    def check_chunk(start):
        chunk = records[start:start + chunk_size]
        valid, skipped = [], []
        for offset, record in enumerate(chunk):
            index = start + offset
            reason = deps.train.validate_record(
                record, local.processor, target_config, cfg, Path(cfg["train_data"]).parent,
            )
            if reason is None:
                valid.append(index)
            else:
                skipped.append({"index": index, "id": record["id"], "reason": reason})
        return valid, skipped, len(chunk)

    valid, skipped = [], []
    starts = iter(range(0, len(records), chunk_size))
    with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        pending = set()
        try:
            # Bound outstanding work, including on errors or Ctrl+C.
            for _ in range(workers * 2):
                start = next(starts, None)
                if start is None:
                    break
                pending.add(pool.submit(check_chunk, start))
            with deps.tqdm(total=len(records), desc="Parallel VLM validation", unit="sample") as bar:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        accepted, rejected, count = future.result()
                        valid.extend(accepted)
                        skipped.extend(rejected)
                        bar.update(count)
                        start = next(starts, None)
                        if start is not None:
                            pending.add(pool.submit(check_chunk, start))
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return {"valid": sorted(valid), "skipped": sorted(skipped, key=lambda row: row["index"])}


def prepare(args):
    deps = load_dependencies()
    cfg = deps.train.load_config(args.config)
    # These modes resolve their block size/checkpoint inside the training runner.
    if cfg.get("reader_only") or cfg.get("init_checkpoint"):
        raise ValueError("Preprocessing requires a resolved from-scratch config (no reader_only/init_checkpoint)")
    cfg.update(
        train_data=str(Path(args.train_data).expanduser().resolve()),
        target_model=str(Path(args.target_model).expanduser().resolve()),
        debug_max_samples=args.max_samples, debug_max_steps=None,
    )
    data = Path(cfg["train_data"])
    cfg["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    records = list(deps.train.read_records(data, args.max_samples or None))
    if not records:
        raise ValueError("No input records")
    print(f"VLM preflight: {len(records)} records; image_max_tokens={cfg.get('image_max_tokens')}", flush=True)
    if args.reuse_validation_from:
        directory = Path(args.reuse_validation_from).expanduser().resolve()
        deps.train.reuse_validation(records, cfg, directory)
        return directory

    cache_root = Path(args.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(signature(cfg), sort_keys=True).encode()).hexdigest()
    # Both Full and Pivot1 use the same key; only the lock owner reads images.
    print(f"Waiting for validation cache lock: {key[:12]}", flush=True)
    with (cache_root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not args.force:
            cached = find_cache(records, cfg, [cache_root, *args.search_root], deps.train)
            if cached is not None:
                return cached

        started = time.monotonic()
        target_config = deps.AutoConfig.from_pretrained(cfg["target_model"], local_files_only=True)
        processor = deps.AutoProcessor.from_pretrained(cfg["target_model"], local_files_only=True)
        if cfg.get("image_max_tokens") is not None:
            cfg["effective_image_processor_size"] = deps.configure(
                processor, target_config, int(cfg["image_max_tokens"]),
            )
        del processor
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        workers = min(args.workers, available)
        print(f"Validating with {workers} CPU threads; no Target weights loaded", flush=True)
        eligible = parallel_validate(records, cfg, target_config, deps, workers, args.chunk_size)
        if not eligible["valid"]:
            raise ValueError("No eligible training records")

        temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=cache_root))
        directory = cache_root / (key + "_" + temporary.name.removeprefix(".building-"))
        try:
            cfg.update(
                output_dir=str(directory), eligible_samples=len(eligible["valid"]),
                eligible_indices_sha256=hashlib.sha256(json.dumps(eligible["valid"]).encode()).hexdigest(),
                validation_workers=workers, validation_elapsed_s=time.monotonic() - started,
            )
            (temporary / "skipped_training.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in eligible["skipped"]),
            )
            (temporary / "effective_config.json").write_text(json.dumps(cfg, indent=2))
            deps.train.reuse_validation(records, cfg, temporary)
            # Publish only a complete cache. Interrupted runs cannot be reused.
            temporary.rename(directory)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        atomic_write(directory / "reuse_validation.env", "export REUSE_VALIDATION_FROM=" + shlex.quote(str(directory)) + "\n")
        return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "train-data", "target-model", "cache-root"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--result-file", type=Path, help="atomically write the selected cache path for a launcher")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reuse-validation-from")
    group.add_argument("--force", action="store_true", help="recheck images even if a compatible cache exists")
    args = parser.parse_args(argv)
    if args.max_samples < 0 or args.workers < 1 or args.chunk_size < 1:
        parser.error("max-samples must be >=0; workers and chunk-size must be >=1")
    directory = prepare(args)
    if args.result_file is not None:
        atomic_write(args.result_file, str(directory) + "\n")
    print("export REUSE_VALIDATION_FROM=" + shlex.quote(str(directory)), flush=True)
    return directory


if __name__ == "__main__":
    main()
