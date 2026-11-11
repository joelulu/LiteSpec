"""Eight datasets, arbitrary DFlash checkpoints, GPU queues and frozen inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import defaultdict, deque
import re
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PLAN = (
    ("gsm8k", "text"), ("charxiv_val", "image"), ("textvqa", "image"),
    ("flickr30k", "image"), ("hr_bench_4k", "image"), ("hr_bench_8k", "image"),
    ("VideoDetailCaption", "video"), ("Video-MME", "video"),
)
FIELDS = ("dataset", "kind", "gpu", "model", "samples", "batch_size", "committed_length",
          "first_hit_rate", "tok_s", "speedup", "matched", "requests", "peak_gib")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def fingerprint(path):
    path = Path(path).expanduser().resolve(strict=True)
    config = path / "config.json"
    weights = sorted([*path.glob("*.safetensors"), *path.glob("*.bin")])
    if not config.is_file() or not weights:
        raise FileNotFoundError(f"Incomplete checkpoint (config/weights required): {path}")
    return dict(path=str(path), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                weights=[dict(name=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in weights])


def frozen_selection(args):
    """Reuse normalized JSONL/video snapshots without reapplying prompt readers."""
    source = str(args.reuse_selection_from)
    if source == 'latest-exp1':
        candidates = list((args.workspace.expanduser()/'runs').glob(
            'dflash_exp1_10w_5ep_*/eval_all_b1_n32/manifest.json'))
        if not candidates:
            raise FileNotFoundError('No experiment-1 evaluation manifest found; pass --reuse-selection-from /path/to/eval_all_b1_n32')
        manifest_file = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    else:
        path = Path(source).expanduser().resolve(strict=True)
        manifest_file = path/'manifest.json' if path.is_dir() else path
    manifest_file = manifest_file.resolve(strict=True)
    prior = json.loads(manifest_file.read_text())
    for key in ('samples', 'video_samples', 'max_new_tokens', 'seed', 'image_max_tokens', 'frames', 'max_pixels_per_frame'):
        if prior['settings'].get(key) != getattr(args, key):
            raise ValueError(f'Frozen selection setting differs: {key}; choose the matching source evaluation')
    jobs = {job['dataset']: job for job in prior['jobs']}
    packets, failures = {}, []
    for name, kind in PLAN:
        try:
            job = jobs[name]
            count = args.video_samples if kind == 'video' else args.samples
            if job['kind'] != kind or job['samples'] != count:
                raise ValueError('source modality/sample count differs')
            path = Path(job['snapshot'])
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != job['input_sha256']:
                raise ValueError('frozen input hash changed')
            if kind == 'video':
                selection = json.loads(raw)
                if set(selection) != {name}:
                    raise ValueError('video snapshot must contain its own dataset only')
                records = selection[name]
                packets[name] = records
            else:
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
                if any(not isinstance(row.get('question'), str) or not row['question'].strip() for row in records):
                    raise ValueError('normalized snapshot requires nonempty question strings')
                packets[name] = dict(source=job.get('source',str(path)), samples=count,
                                     serialized=raw.decode('utf-8'), sha256=digest)
            if len(records) != count or len({row['id'] for row in records}) != count:
                raise ValueError('frozen snapshot has incorrect/duplicate sample IDs')
        except Exception as exc:
            failures.append(f'{name}: {type(exc).__name__}: {exc}')
    if failures:
        raise ValueError('Frozen dataset preparation failed before loading models:\n'+'\n'.join(failures))
    metadata = dict(manifest=str(manifest_file),
                    manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest())
    return packets, metadata


def prepare(args):

    root = args.output_dir.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Choose a fresh evaluation output directory: {root}")
    group = args.run_group.expanduser().resolve(strict=True)
    selections = getattr(args, "checkpoint", None) or [(name, group / name / "last") for name in ("full", "pivot1")]
    labels = [label for label, path in selections]
    if len(set(labels)) != len(labels) or any(label == "ar" or not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in labels):
        raise ValueError("Checkpoint labels must be unique, safe names other than ar")
    models = {label: fingerprint(path) for label, path in selections}
    target = fingerprint(args.target_model)
    for label, item in models.items():
        config = json.loads((Path(item["path"]) / "config.json").read_text())
        context = config.get("vlm_config", {}).get("draft_context", "full")
        known = {name: name for name in ("full", "recent128", "pivot1", "pivot5", "pivot12")}
        known.update(pivot5_B="pivot5", pivot5_C="pivot5")
        if config.get("architectures") != ["DFlashDraftModel"] or context != known.get(label, context):
            raise ValueError(f"Checkpoint {label} has the wrong architecture/context: {item['path']}")
        if label in ("pivot5_B", "pivot5_C"):
            method = config.get("dflash_config", {})
            if method.get("context_projection") != "per_layer" or method.get("position_layer_bias", False) != (label == "pivot5_C"):
                raise ValueError(f"Checkpoint {label} has the wrong layer adapter: {item['path']}")

    # Explicitly override inherited PROMPT_FIELD=question. Existing reference
    # policy reads prompt, otherwise the first conversation content.
    os.environ["PROMPT_FIELD"] = args.prompt_field
    os.environ["DATA_DIR"] = str(args.data_dir.expanduser())
    os.environ["IMAGE_ROOT"] = str((args.image_root or args.data_dir / "imgs").expanduser())
    if args.text_data is not None:
        os.environ["TEXT_DATA"] = str(args.text_data.expanduser())
    selection_reuse = None
    if getattr(args, 'reuse_selection_from', None):
        packets, selection_reuse = frozen_selection(args)
        print(f"Reusing frozen evaluation inputs: {selection_reuse['manifest']}", flush=True)
    else:
        from run_vlm_matrix import select_data as select_images
        from video_benchmark_data import select_data as select_videos
        packets, failures = {}, []
        for name, kind in PLAN:
            try:
                if kind == "video":
                    packets[name] = select_videos(args.video_data_root, [name], args.video_samples, args.seed)[name]
                else:
                    selected = select_images(argparse.Namespace(
                        tests=["text-only" if name == "gsm8k" else name], samples=args.samples, seed=args.seed,
                    ))
                    packets[name] = selected[name]
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if failures:
            raise ValueError("Dataset preparation failed before loading models:\n" + "\n".join(failures))

    vision = json.loads((Path(target["path"]) / "config.json").read_text())["vision_config"]
    image_pixels = args.image_max_tokens * (vision["patch_size"] * vision["spatial_merge_size"]) ** 2
    (root / "inputs").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "results").mkdir()
    jobs = []
    for index, (name, kind) in enumerate(PLAN):
        gpu = args.gpus[index % len(args.gpus)]
        sample_count = args.video_samples if kind == "video" else args.samples
        if kind == "video":
            snapshot = root / "inputs" / (name + ".json")
            write_json(snapshot, {name: packets[name]})
        else:
            snapshot = root / "inputs" / (name + ".jsonl")
            snapshot.write_text(packets[name]["serialized"])
        job = dict(dataset=name, kind=kind, gpu=gpu, samples=sample_count,
                   snapshot=str(snapshot), output=str(root / "results" / name), log=str(root / "logs" / (name + ".log")))
        job["models"] = ["ar", *models]
        job["input_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        if kind != "video":
            job["source"] = packets[name]["source"]
        job["command"] = command(args, job, models, target["path"], image_pixels)
        jobs.append(job)
    manifest = dict(run_group=str(group), target=target, checkpoints=models, selection_reuse=selection_reuse,
                    settings=vars(args), image_max_pixels=image_pixels, jobs=jobs)
    write_json(root / "manifest.json", manifest)
    return root, jobs


def command(args, job, models, target, image_pixels):
    video = job["kind"] == "video"
    cmd = [sys.executable, str(HERE / ("benchmark_video.py" if video else "benchmark_vlm_static_batch.py")),
           "--target-model", str(target), "--output-dir", job["output"],
           "--batch-sizes", str(args.batch_size), "--max-new-tokens", str(args.max_new_tokens),
           "--seed", str(args.seed), "--warmup", str(args.warmup)]
    for label, item in models.items():
        cmd += ["--checkpoint", label, item["path"]]
    if video:
        cmd += ["--workspace", str(args.workspace), "--data-root", str(args.video_data_root),
                "--datasets", job["dataset"], "--selection-file", job["snapshot"],
                "--samples", str(job["samples"]), "--frames", str(args.frames),
                "--max-pixels-per-frame", str(args.max_pixels_per_frame)]
    else:
        cmd += ["--data", job["snapshot"], "--dataset-name", job["dataset"],
                "--max-samples", str(job["samples"]), "--image-max-pixels", str(image_pixels),
                "--cpu-inputs", "--attention-backend", "sdpa", "--state-mode", "per_token",
                "--visual-condition", "correct"]
    return cmd


def normalized(job, row):
    video = job["kind"] == "video"
    return dict(dataset=job["dataset"], kind=job["kind"], gpu=job["gpu"], model=row["model"],
                samples=job["samples"], batch_size=row["batch_size"],
                committed_length=row["mean_committed_length"], first_hit_rate=row.get("first_candidate_hit_rate"),
                tok_s=row["decode_tok_s" if video else "output_tokens_per_second"],
                speedup=row["raw_decode_ratio_vs_ar" if video else "raw_throughput_speedup_vs_ar"],
                matched=row["matched"], requests=row["requests"], peak_gib=row["peak_allocated_gib"])


def read_summary(job, batch_size):
    rows = json.loads((Path(job["output"]) / "summary.json").read_text())
    labels = [row["model"] for row in rows]
    expected = job.get("models", ["ar", "full", "pivot1"])
    if len(labels) != len(expected) or set(labels) != set(expected):
        raise ValueError(f"Expected completed {expected} rows, got {labels}")
    if any(row["batch_size"] != batch_size or row["requests"] != job["samples"] * batch_size for row in rows):
        raise ValueError("Summary batch/sample counts differ from the requested job")
    # Token mismatches remain visible but do not hide measured throughput.
    return [normalized(job, row) for row in rows]


def report(root, jobs, statuses):
    rows = []
    for job in jobs:
        if statuses.get(job["dataset"], {}).get("status") == "success":
            data = json.loads((Path(job["output"]) / "summary.json").read_text())
            rows.extend(normalized(job, row) for row in data)
    write_json(root / "status.json", statuses)
    write_json(root / "comparison.json", rows)
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# DFlash evaluation", "",
             "B is static batch size. Ratios use same-GPU AR; timing is decode-only.",
             "Video-MME uses the existing video-description protocol, not official QA accuracy.", "",
             "| Dataset | GPU | Model | B | Committed | First hit | tok/s | Speedup | AR match | Peak GiB |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        length = "—" if row["committed_length"] is None else f"{row['committed_length']:.3f}"
        hit = "—" if row["first_hit_rate"] is None else f"{row['first_hit_rate']:.1%}"
        lines.append(f"| {row['dataset']} | {row['gpu']} | {row['model']} | {row['batch_size']} | {length} | {hit} | "
                     f"{row['tok_s']:.2f} | {row['speedup']:.3f}x | {row['matched']}/{row['requests']} | {row['peak_gib']:.3f} |")
    for name, status in statuses.items():
        if status["status"] != "success":
            lines += ["", f"{name}: {status['status']}; see logs/{name}.log"]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")


def execute(root, jobs, batch_size):
    processes, lock, stop = set(), threading.Lock(), threading.Event()
    statuses = {}

    def run(job):
        started = time.monotonic()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job["gpu"]),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        with Path(job["log"]).open("w") as log:
            with lock:
                if stop.is_set():
                    return dict(status="cancelled", returncode=130)
                process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT,
                                           env=env, cwd=HERE.parent, start_new_session=True)
                processes.add(process)
            try:
                code = process.wait()
            finally:
                with lock:
                    processes.discard(process)
        result = dict(status="success" if code == 0 else "failed", returncode=code,
                      elapsed_s=time.monotonic() - started, log=job["log"])
        if code == 0:
            try:
                read_summary(job, batch_size)
            except Exception as exc:
                result.update(status="invalid_summary", returncode=1, error=str(exc))
        return result

    queues = defaultdict(deque)
    for job in jobs:
        queues[job["gpu"]].append(job)
    pool = ThreadPoolExecutor(max_workers=max(1, len(queues)))
    pending = {}
    try:
        for queue in queues.values():
            job = queue.popleft()
            pending[pool.submit(run, job)] = job
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    status = future.result()
                except Exception as exc:
                    status = dict(status="failed", returncode=1, error=str(exc), log=job["log"])
                statuses[job["dataset"]] = status
                print(f"{job['dataset']} GPU {job['gpu']}: {status['status']}; {job['log']}", flush=True)
                report(root, jobs, statuses)
                queue = queues[job["gpu"]]
                if queue:
                    next_job = queue.popleft()
                    pending[pool.submit(run, next_job)] = next_job
    except KeyboardInterrupt:
        stop.set()
        with lock:
            active = list(processes)
        for process in active:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for future in pending:
            future.cancel()
        for job in jobs:
            statuses.setdefault(job["dataset"], dict(status="interrupted", returncode=130, log=job["log"]))
        report(root, jobs, statuses)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return 0 if all(value["status"] == "success" for value in statuses.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-group", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", nargs=2, action="append", metavar=("LABEL", "PATH"),
                        help="repeat for arbitrary checkpoints; defaults to Full/Pivot1")
    parser.add_argument('--reuse-selection-from',
                        help='reuse prior frozen inputs: evaluation directory/manifest or latest-exp1; skips raw prompt readers')
    parser.add_argument("--workspace", type=Path, default=HERE.parent.parent)
    parser.add_argument("--target-model", type=Path, default=Path.home() / "LMM/lmm_model/Qwen3.5-4B")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "LMM/liyt/datasets/jsonl_datas")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--text-data", type=Path)
    parser.add_argument("--video-data-root", type=Path, default=Path.home() / "LMM/liyt/datasets/eval_bench")
    parser.add_argument("--gpus", nargs="+", default=list(map(str, range(8))))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--video-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--image-max-tokens", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--max-pixels-per-frame", type=int, default=200704)
    parser.add_argument("--prompt-field", choices=["reference", "prompt", "question", "first-content"], default="reference")
    parser.add_argument("--dry-run", action="store_true", help="freeze inputs/commands without starting model processes")
    args = parser.parse_args(argv)
    if not 1 <= len(args.gpus) <= 8 or len(set(args.gpus)) != len(args.gpus):
        parser.error("one to eight distinct GPU selectors required")
    if min(args.samples, args.video_samples, args.batch_size, args.max_new_tokens,
           args.warmup, args.image_max_tokens, args.max_pixels_per_frame) < 1 or args.frames < 2 or args.frames % 2:
        parser.error("positive budgets, warmup >=1 and an even frame count >=2 required")
    root, jobs = prepare(args)
    for job in jobs:
        print(f"GPU {job['gpu']}: {job['dataset']}, {job['samples']} samples; {','.join(job['models'])}", flush=True)
        print(shlex.join(job["command"]), flush=True)
    print(f"Results: {root}\nMonitor: tail -f {root}/logs/*.log", flush=True)
    if args.dry_run:
        return 0
    result = execute(root, jobs, args.batch_size)
    print((root / "comparison.md").read_text(), flush=True)
    print(f"Results: {root}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
