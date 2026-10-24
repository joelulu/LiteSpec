"""GPU-affine static HF evaluation queue. Does not implement online concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from reference_vlm_data import DATASETS, dataset_path, load_reference_records

MODELS = ("dlite_vlm_100k", "dflash_vlm_100k", "dflash_text_20k", "dlite_original",
          "dlite_vlm_20k", "dflash_vlm_20k", "dspark_vlm_20k")
MODES = ("per_token", "compact_replay", "full_target_replay")
STOP = threading.Event()
PROCESSES, PRINT_LOCK, PROCESS_LOCK = set(), threading.Lock(), threading.RLock()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def message(text):
    with PRINT_LOCK:
        print(text, flush=True)


def checkpoints(workspace, selected, overrides=None):
    overrides = overrides or {}
    old = workspace/"runs/vlm_three_3ep_20261021_045040_2983971"
    new = workspace/"runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049"
    defaults = dict(dlite_vlm_100k=new/"dlite/last", dflash_vlm_100k=new/"dflash/last",
        dlite_original=Path.home()/"LMM/lmm_model/Qwen3.5-4B-DLite",
        dlite_vlm_20k=old/"dlite/last", dflash_vlm_20k=old/"dflash/last", dspark_vlm_20k=old/"dspark/last")
    env_names = dict(dlite_vlm_100k="DLITE_VLM", dflash_vlm_100k="DFLASH_VLM", dflash_text_20k="DFLASH_TEXT",
        dlite_original="ORIGINAL_DLITE", dlite_vlm_20k="DLITE_3EP", dflash_vlm_20k="DFLASH_3EP", dspark_vlm_20k="DSPARK_3EP")
    result = {}
    for label in selected:
        if label not in overrides and label not in env_names:
            raise ValueError(f"unknown model {label}; supply --checkpoint LABEL PATH")
        value = overrides.get(label) or os.environ.get(env_names.get(label, ""))
        if not value and label == "dflash_text_20k":
            candidates = sorted(p for p in (workspace/"runs").glob("dflash_text_20k_3ep_*/last") if (p/"config.json").is_file())
            if len(candidates) != 1:
                raise ValueError(f"Set DFLASH_TEXT to the intended text checkpoint; found {len(candidates)} candidates: {candidates}")
            value = candidates[0]
        path = Path(value or defaults[label]).expanduser().resolve()
        config = json.loads((path/"config.json").read_text())
        result[label] = dict(path=str(path), config_sha256=hashlib.sha256((path/"config.json").read_bytes()).hexdigest(),
            architecture=config.get("architectures"), block_size=config.get("block_size"),
            num_draft_layers=config.get("num_hidden_layers"), gdn_reader_config=config.get("gdn_reader_config"),
            weight_files=[dict(name=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                          for p in sorted(path.iterdir()) if p.suffix in (".safetensors", ".bin")])
        effective = path.parent/"effective_config.json"
        if effective.is_file():
            training = json.loads(effective.read_text())
            training = training.get("config", training)
            result[label]["training"] = {k:training.get(k) for k in ("eligible_samples", "num_epochs", "batch_size", "accumulation_steps", "world_size", "block_size")}
    return result


def select_data(args):
    import random
    directory = Path(os.environ.get("DATA_DIR", str(Path.home()/"LMM/liyt/datasets/jsonl_datas"))).expanduser()
    image_root = Path(os.environ.get("IMAGE_ROOT", str(directory/"imgs"))).expanduser()
    selected = {}
    for name in args.tests:
        if name == "text-only":
            source = Path(os.environ.get("TEXT_DATA", str(Path.home()/"LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main-0708-new/eval_datasets/gsm8k.jsonl"))).expanduser().resolve()
            raw = [(i, json.loads(line)) for i, line in enumerate(source.read_text().splitlines()) if line.strip()]
            if len(raw) < args.samples:
                raise ValueError(f"need {args.samples} records in {source}")
            records = []
            for i, row in random.Random(args.seed).sample(raw, args.samples):
                turns = row.get("turns")
                question = row.get("question")
                if turns is not None:
                    if not isinstance(turns, list) or len(turns) != 1:
                        raise ValueError("GSM8K evaluation requires one user turn")
                    question = turns[0]
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"missing question/one turn at {source}:{i+1}")
                records.append(dict(id=f"gsm8k:{i}", question=question, source_row=i))
            case = "gsm8k"
        else:
            source = dataset_path(directory, name)
            records, _ = load_reference_records(source, args.samples, args.seed, image_root, True,
                os.environ.get("PROMPT_FIELD", "reference"))
            if len(records) != args.samples:
                raise ValueError(f"need {args.samples} records in {source}")
            case = name
        serialized = "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records)
        selected[case] = dict(source=str(source), sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                              samples=args.samples, serialized=serialized)
    return selected


def inventory(gpus):
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"], text=True)
    devices = []
    for line in text.splitlines():
        index, uuid, name, memory = (p.strip() for p in line.split(",", 3))
        devices.append(dict(index=index, uuid=uuid, name=name, memory_mib=memory))
    result = []
    for gpu in gpus:
        matches = [d for d in devices if gpu in (d["index"], d["uuid"])]
        if len(matches) != 1:
            raise ValueError(f"GPU not found: {gpu}")
        result.append(dict(matches[0], selector=gpu))
    if len({d["uuid"] for d in result}) != len(result):
        raise ValueError("GPU selectors refer to the same physical GPU")
    if len({d["name"] for d in result}) != 1:
        raise ValueError("use the same GPU model for the benchmark matrix")
    return result


def run_process(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        with PROCESS_LOCK:
            PROCESSES.add(process)
        try:
            code = process.wait()
        finally:
            with PROCESS_LOCK:
                PROCESSES.discard(process)
    if STOP.is_set():
        return "interrupted", code
    if code == 0:
        return "success", code
    tail = log_path.read_text(errors="replace")[-64000:].lower()
    oom = any(s in tail for s in ("cuda out of memory", "cuda error: out of memory", "torch.outofmemoryerror"))
    return ("oom" if oom else "failed"), code


def cell(args, job, gpu, mode, label, checkpoint=None, ar=None):
    directory = args.output_dir/"jobs"/job/ ("ar" if label == "ar" else f"{mode}/{label}")
    status_file = directory/"status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    if status.get("status") == "success" and (directory/"result/summary.json").is_file():
        return status
    if status.get("status") == "oom" and not args.retry_oom:
        return status
    if STOP.is_set():
        return dict(status="interrupted")
    attempt = int(status.get("attempt", 0))+1
    output = directory/f"attempt_{attempt}"
    directory.mkdir(parents=True, exist_ok=True)
    case, batch = job.rsplit("_b", 1)
    command = [sys.executable, str(Path(__file__).with_name("benchmark_vlm_static_batch.py")),
        "--target-model", args.target_model, "--data", str(args.output_dir/"inputs"/f"{case}.jsonl"),
        "--dataset-name", case, "--data-split", "unspecified", "--output-dir", str(output),
        "--batch-sizes", batch, "--max-samples", str(args.samples), "--max-new-tokens", str(args.max_new_tokens),
        "--seed", str(args.seed), "--warmup", str(args.warmup), "--attention-backend", args.attention_backend,
        "--state-mode", mode, "--cpu-inputs"]
    if checkpoint:
        command += ["--checkpoint", label, checkpoint]
    if ar:
        command += ["--ar-reference", str(ar)]
    for option, value in args.image_options.items():
        command += [option, value]
    if args.profile_decode:
        command += ["--profile-decode", "--profile-samples", str(args.profile_samples)]
    device = getattr(args, "device_selectors", {}).get(gpu, gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
    state = dict(status="running", gpu=gpu, mode=mode, model=label, attempt=attempt, command=command,
                 started=time.time(), output=str(output))
    write_json(status_file, state)
    message(f"GPU {gpu}: {job} {mode}/{label} started")
    result, code = run_process(command, directory/f"attempt_{attempt}.log", env)
    state.update(status=result, returncode=code, finished=time.time())
    if result == "success":
        expected = [r for r in json.loads((output/"summary.json").read_text()) if r["model"] == label]
        if len(expected) != 1:
            state.update(status="failed", error="missing or duplicate completed summary row")
        else:
            link = directory/"result"
            temporary = directory/"result.new"
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(output.name, target_is_directory=True)
            temporary.replace(link)
    write_json(status_file, state)
    message(f"GPU {gpu}: {job} {mode}/{label}: {state['status']}")
    return state


def worker(args, gpu, jobs, mode, models):
    for job in jobs:
        if STOP.is_set():
            return
        base = cell(args, job, gpu, "full_target_replay", "ar")
        if base["status"] != "success":
            message(f"GPU {gpu}: {job} blocked by AR {base['status']}; other jobs continue")
            continue
        ar = args.output_dir/"jobs"/job/"ar/result"
        for label, config in models.items():
            if STOP.is_set():
                return
            cell(args, job, gpu, mode, label, config["path"], ar)


def summarize(root):
    manifest = json.loads((root/"manifest.json").read_text())
    rows, baselines = [], []
    for job, gpu in manifest["assignments"].items():
        case, batch = job.rsplit("_b", 1)
        baseline_path = root/"jobs"/job/"ar/result/summary.json"
        base = next((r for r in json.loads(baseline_path.read_text()) if r["model"] == "ar"), None) if baseline_path.exists() else None
        if base:
            baselines.append(dict(dataset=case, batch_size=int(batch), gpu=gpu,
                decode_tok_s=base["decode_tokens_per_second"], peak_gib=base["peak_allocated_gib"]))
        for mode in manifest["state_modes"]:
            for label in manifest["models"]:
                directory = root/"jobs"/job/mode/label
                state = json.loads((directory/"status.json").read_text()) if (directory/"status.json").exists() else {}
                row = dict(dataset=case, batch_size=int(batch), gpu=gpu, model=label, state_mode=mode,
                    status=state.get("status", "pending"), accept_length=None, decode_tok_s=None, speedup=None,
                    peak_gib=None, record_mib=None, target_calls_per_round=None, profile_ms_per_round=None,
                    valid_token_match=None, draft_ms=None, gdn_state_prepare_ms=None,
                    verify_ms=None, target_replay_ms=None, state_commit_ms=None,
                    token_exact_vs_per_token=None, accepts_equal_vs_per_token=None)
                if state.get("status") == "success":
                    result = directory/"result"
                    summary = next(r for r in json.loads((result/"summary.json").read_text()) if r["model"] == label)
                    row.update(accept_length=summary["mean_committed_length"], decode_tok_s=summary["decode_tokens_per_second"],
                        speedup=summary["decode_tokens_per_second"]/base["decode_tokens_per_second"],
                        peak_gib=summary["peak_allocated_gib"], record_mib=summary["peak_transaction_record_mib"],
                        valid_token_match=summary.get("valid_token_match"))
                    profiles = result/"decode_profile.json"
                    if profiles.exists():
                        profile = next(r for r in json.loads(profiles.read_text()) if r["model"] == label)
                        row.update(target_calls_per_round=profile["target_calls_per_round"], profile_ms_per_round=profile["ms_per_round"])
                        stages = profile["stage_seconds"]
                        row.update(draft_ms=1000*stages.get("draft",0), gdn_state_prepare_ms=1000*stages.get("gdn_state_prepare",0))
                        row.update(verify_ms=1000*stages.get("target_verify",0), target_replay_ms=1000*stages.get("target_replay",0),
                            state_commit_ms=1000*(stages.get("state_select",0)+stages.get("gdn_state_replay",0)))
                    reference = root/"jobs"/job/"per_token"/label/"result"/f"{label}_b{batch}.jsonl"
                    if reference.exists():
                        def records(p):
                            return {r["sample_id"]:(r["result"]["token_ids"], r["result"]["accept_lengths"])
                                    for r in map(json.loads, p.read_text().splitlines())}
                        a, b = records(result/f"{label}_b{batch}.jsonl"), records(reference)
                        row.update(token_exact_vs_per_token={k:v[0] for k,v in a.items()} == {k:v[0] for k,v in b.items()},
                                   accepts_equal_vs_per_token={k:v[1] for k,v in a.items()} == {k:v[1] for k,v in b.items()})
                elif base is None and row["status"] == "pending":
                    row["status"] = "blocked_ar"
                rows.append(row)
    with (root/"matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(root/"ar_baselines.json", baselines)
    lines = [f"# {len(manifest['models'])}-draft static-batch evaluation", "",
        "Each cell: speedup / mean committed length. Decode-only, against same-GPU/same-batch AR.",
        "Each prompt is replicated B times; not online continuous batching. OOM is not replaced with a smaller batch.",
        "Native HF vs new state modes also changes the verification kernel; compact replay is not full upstream ReplaySSM.",
        "A dagger (†) marks token mismatch against AR; those ratios are descriptive, not a lossless-speedup claim.",
        "Reader compute is included in draft_ms; gdn_state_prepare_ms covers committed-cache state access only.", ""]
    for mode in manifest["state_modes"]:
        for case in manifest["datasets"]:
            batches = manifest["batch_sizes"]
            lines += [f"## {case} — {mode}", "", "| Model | "+" | ".join(f"B={b}" for b in batches)+" |",
                      "|---|"+"---:|"*len(batches)]
            lines.append("| ar | "+" | ".join("1.000× / —" if any(r["dataset"] == case and r["batch_size"] == b for r in baselines)
                else "unavailable" for b in batches)+" |")
            for label in manifest["models"]:
                cells = []
                for batch in batches:
                    r = next(r for r in rows if r["dataset"] == case and r["batch_size"] == batch and r["model"] == label and r["state_mode"] == mode)
                    cells.append((f"{r['speedup']:.3f}× / {r['accept_length']:.3f}" + (" †" if r["valid_token_match"] is False else "")) if r["status"] == "success" else r["status"])
                lines.append(f"| {label} | "+" | ".join(cells)+" |")
            lines.append("")
    (root/"matrix.md").write_text("\n".join(lines)+"\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0,1,2,3,4,5,6,7"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--checkpoint", nargs=2, action="append", default=[], metavar=("LABEL", "PATH"))
    parser.add_argument("--tests", nargs="+", choices=["text-only", *DATASETS], default=["text-only", *DATASETS])
    parser.add_argument("--state-modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1,16,64])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--attention-backend", choices=["auto","sdpa","eager","flash_attention_2"], default="auto")
    parser.add_argument("--profile-decode", action="store_true")
    parser.add_argument("--profile-samples", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-oom", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    overrides = dict(args.checkpoint)
    if len(overrides) != len(args.checkpoint):
        parser.error("duplicate checkpoint label")
    if any(not label or label == "ar" or any(not (c.isalnum() or c in "-_") for c in label) for label in overrides):
        parser.error("checkpoint labels must contain letters/digits/hyphens/underscores and cannot be ar")
    args.models = args.models or list(overrides) or list(MODELS)
    args.gpus = [s for token in args.gpus for s in token.replace(",", " ").split()]
    for values in (args.gpus, args.models, args.tests, args.state_modes, args.batch_sizes):
        if not values or len(values) != len(set(values)):
            parser.error("selectors must be nonempty and unique")
    if min(args.samples, args.max_new_tokens, args.profile_samples, *args.batch_sizes) < 1 or args.warmup < 0:
        parser.error("positive sample/token/batch limits and nonnegative warmup required")
    args.workspace, args.output_dir = args.workspace.expanduser().resolve(), args.output_dir.expanduser().resolve()
    args.target_model = str(Path(os.environ.get("TARGET_MODEL", str(Path.home()/"LMM/lmm_model/Qwen3.5-4B"))).expanduser().resolve())
    target_hash = hashlib.sha256((Path(args.target_model)/"config.json").read_bytes()).hexdigest()
    models, data = checkpoints(args.workspace, args.models, overrides), select_data(args)
    args.image_options = {flag:os.environ[env] for flag,env in (("--image-max-pixels","IMAGE_MAX_PIXELS"),("--image-min-pixels","IMAGE_MIN_PIXELS"),("--image-max-tokens","IMAGE_MAX_TOKENS")) if os.environ.get(env)}
    if "--image-max-tokens" in args.image_options and len(args.image_options) != 1:
        parser.error("choose IMAGE_MAX_TOKENS or IMAGE_*_PIXELS, not both")
    assignments = {f"{name}_b{batch}":args.gpus[i%len(args.gpus)]
        for i,(name,batch) in enumerate((name,batch) for name in data for batch in args.batch_sizes)}
    identity = dict(version=1, target_model=args.target_model, target_config_sha256=target_hash,
        models=models, datasets={k:{a:b for a,b in v.items() if a != "serialized"} for k,v in data.items()},
        assignments=assignments, gpus=args.gpus, batch_sizes=args.batch_sizes, state_modes=args.state_modes,
        samples=args.samples, max_new_tokens=args.max_new_tokens, seed=args.seed, warmup=args.warmup,
        attention_backend=args.attention_backend, image_options=args.image_options,
        profile_decode=args.profile_decode, profile_samples=args.profile_samples)
    repo = Path(__file__).resolve().parents[1]
    identity["evaluation_code_sha256"] = {p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in
        ("tests/benchmark_vlm_static_batch.py", "litespec/vlm/static_batch.py", "litespec/vlm/gdn_state.py", "litespec/vlm/gdn_state_triton.py", "litespec/vlm/gdn_reader.py", "litespec/vlm/baselines.py", "litespec/modeling/draft/dlite.py", "litespec/modeling/draft/dflash.py")}
    if args.output_dir.exists():
        if not args.resume:
            parser.error("RESULT_DIR exists; use RESUME=1 / --resume")
        prior = json.loads((args.output_dir/"manifest.json").read_text())
        if prior != identity:
            parser.error("resume settings/checkpoints/data/GPU assignments differ; choose a fresh RESULT_DIR")
    else:
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir/"manifest.json", identity)
    lock = (args.output_dir/"queue.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another scheduler is already using this RESULT_DIR") from exc
    for name, info in data.items():
        path = args.output_dir/"inputs"/f"{name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        if path.exists() and path.read_text() != info["serialized"]:
            raise ValueError(f"evaluation snapshot changed: {path}")
        path.write_text(info["serialized"])
    message(f"Results: {args.output_dir}; {len(assignments)} GPU-affine jobs, {len(models)} drafts, stages={args.state_modes}")
    if args.dry_run:
        message(json.dumps(identity, indent=2)); return
    devices = inventory(args.gpus)
    inventory_file = args.output_dir/"gpu_inventory.json"
    if inventory_file.exists() and json.loads(inventory_file.read_text()) != devices:
        raise ValueError("physical GPUs changed since previous run")
    write_json(inventory_file, devices)
    args.device_selectors = {d["selector"]:d["uuid"] for d in devices}
    def cancel():
        STOP.set()
        with PROCESS_LOCK:
            active = list(PROCESSES)
        for process in active:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    def interrupted(signum, frame):
        cancel()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    # Validate GPU math once on each participating device before large model loads.
    for gpu in args.gpus:
        status, _ = run_process([sys.executable, str(Path(__file__).with_name("check_gdn_state_cuda.py")), "--width", "16"],
            args.output_dir/f"cuda_check_{gpu}.log", dict(os.environ, CUDA_VISIBLE_DEVICES=args.device_selectors[gpu]))
        if status != "success":
            raise RuntimeError(f"GPU {gpu} state-kernel validation failed; inspect cuda_check log")
    executor = ThreadPoolExecutor(max_workers=len(args.gpus))
    try:
        for mode in args.state_modes:
            message(f"=== Stage {mode} ===")
            futures = [executor.submit(worker, args, gpu, [j for j,g in assignments.items() if g == gpu], mode, models)
                       for gpu in args.gpus]
            for future in as_completed(futures):
                future.result()
            rows = summarize(args.output_dir)
            message(f"Stage {mode} complete; {sum(r['status']=='success' for r in rows)}/{len(rows)} cells successful; matrix.md updated")
    except BaseException:
        cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    rows = summarize(args.output_dir)
    bad = [r for r in rows if r["status"] not in ("success", "oom")]
    message(f"Done: {args.output_dir}/matrix.md; OOM cells={sum(r['status']=='oom' for r in rows)}; other incomplete={len(bad)}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
